"""Re-encode identical complete inputs and evaluate post-encoding pairing nulls."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scus.artifacts import sha256
from scus.config import load_config
from scus.joint_reconstruction import JointModel, load_joint_data
from scus.joint_batch import encoded_cells
from scus.runtime import write_json
from align_pairing_null import load_projector, cosine

MODELS = ['zero_shot', 'adapter_only', 'joint_epoch29']
CONDITIONS = ['correct', 'matched_cell', 'within_cell_gene']


def derangement(n, seed, cell_id):
    if n < 2:
        raise ValueError('At least two common genes required')
    key = int.from_bytes(hashlib.sha256(f'{seed}:{cell_id}:gene-null'.encode()).digest()[:8], 'little')
    order = np.random.default_rng(key).permutation(n)
    result = np.empty(n, dtype=np.int64)
    result[order] = np.roll(order, 1)
    assert np.all(result != np.arange(n))
    return result


def build_pairs(meta, genes, seed=42):
    rng = np.random.default_rng(seed)
    records, indices = [], []
    for i, row in enumerate(meta.itertuples()):
        candidate = np.flatnonzero((meta.cell_type.to_numpy() == row.cell_type) &
                                  (meta['sample'].to_numpy() != row.sample))
        if not len(candidate):
            candidate = np.flatnonzero((meta.cell_type.to_numpy() == row.cell_type) & (np.arange(len(meta)) != i))
        if not len(candidate):
            raise ValueError('No nonself same-type donor')
        donor = int(rng.choice(candidate))
        common, own, other = np.intersect1d(genes[i], genes[donor], return_indices=True)
        permutation = derangement(len(common), seed, row.cell_id)
        indices.append((donor, own, other, permutation))
        records.append(dict(cell_index=i, cell_id=row.cell_id, cell_type=row.cell_type,
                            sample=row.sample, split=row.split, donor_index=donor,
                            donor_cell_id=meta.iloc[donor].cell_id, donor_sample=meta.iloc[donor]['sample'],
                            n_pairs=len(common), input_paired_genes=len(genes[i]),
                            common_genes_sha256=hashlib.sha256(common.tobytes()).hexdigest(),
                            permutation_sha256=hashlib.sha256(permutation.tobytes()).hexdigest()))
    return indices, pd.DataFrame(records)


def build_cross_pairs(meta, genes, seed=42):
    """Keep the original same-type donor, and use a triple gene intersection."""
    previous, manifest = build_pairs(meta, genes, seed)
    records, indices = [], []
    for i, row in enumerate(meta.itertuples()):
        donor = previous[i][0]
        different = meta.cell_type.to_numpy() != row.cell_type
        # Prefer the same sample as the same-type donor, controlling sample source.
        candidates = np.flatnonzero(different & (meta['sample'].to_numpy() == meta.iloc[donor]['sample']))
        if not len(candidates):
            candidates = np.flatnonzero(different & (meta['sample'].to_numpy() != row.sample))
        if not len(candidates):
            candidates = np.flatnonzero(different)
        if not len(candidates):
            raise ValueError('No different-cell-type donor')
        key = int.from_bytes(hashlib.sha256(f'{seed}:{row.cell_id}:cross-type-donor'.encode()).digest()[:8], 'little')
        cross = int(np.random.default_rng(key).choice(candidates))
        common = np.intersect1d(np.intersect1d(genes[i], genes[donor]), genes[cross])
        def locate(g):
            lookup = {int(gene): j for j, gene in enumerate(g)}
            return np.asarray([lookup[int(gene)] for gene in common], dtype=np.int64)
        own, other, cross_pos = [locate(g) for g in [genes[i], genes[donor], genes[cross]]]
        permutation = derangement(len(common), seed, row.cell_id)
        indices.append((donor, cross, own, other, cross_pos, permutation))
        record = manifest.iloc[i].to_dict()
        record.update(previous_two_way_n_pairs=int(record['n_pairs']),n_pairs=len(common),
                      donor_cell_type=meta.iloc[donor].cell_type,cross_donor_index=cross,
                      cross_donor_cell_id=meta.iloc[cross].cell_id,
                      cross_donor_cell_type=meta.iloc[cross].cell_type,
                      cross_donor_sample=meta.iloc[cross]['sample'],
                      common_genes_sha256=hashlib.sha256(common.tobytes()).hexdigest(),
                      permutation_sha256=hashlib.sha256(permutation.tobytes()).hexdigest())
        records.append(record)
    return indices, pd.DataFrame(records)


def selftest():
    meta = pd.DataFrame(dict(cell_id=['a', 'b', 'c', 'd'], cell_type=['x']*4,
                             sample=['s1', 's1', 's2', 's2'], split=['train']*4))
    genes = [np.array([4, 2, 7]), np.array([7, 4, 2, 9]), np.array([2, 7, 4]), np.array([2, 4, 7, 8])]
    ids, records = build_pairs(meta, genes)
    for i, (d, own, other, perm) in enumerate(ids):
        assert i != d and meta.iloc[i]['sample'] != meta.iloc[d]['sample']
        assert np.array_equal(genes[i][own], genes[d][other])
        assert np.all(genes[i][own] != genes[i][own][perm])
        assert np.array_equal(perm, derangement(len(own), 42, meta.iloc[i].cell_id))
    u = np.eye(3, dtype=np.float32)
    assert np.allclose(cosine(u, u), 0)
    assert np.allclose(cosine(u, np.roll(u, 1, axis=0)), 1)
    assert records.n_pairs.eq(3).all()
    cross_meta=pd.concat([meta,meta.assign(cell_id=['e','f','g','h'],cell_type='y')],ignore_index=True)
    cross_genes=genes+[np.array([7,4,11])]*4
    original,_=build_pairs(cross_meta,cross_genes)
    triples,man=build_cross_pairs(cross_meta,cross_genes)
    for i,(d,x,own,other,xpos,perm) in enumerate(triples):
        assert d==original[i][0]
        assert cross_meta.iloc[i].cell_type!=cross_meta.iloc[x].cell_type
        assert cross_meta.iloc[d]['sample']==cross_meta.iloc[x]['sample']
        assert np.array_equal(cross_genes[i][own],cross_genes[d][other])
        assert np.array_equal(cross_genes[i][own],cross_genes[x][xpos])
        assert np.all(perm!=np.arange(len(own)))
    assert man.n_pairs.eq(2).all()
    try: build_cross_pairs(meta,genes)
    except ValueError: pass
    else: raise AssertionError('Single-type cohort accepted')


def plots(out, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    titles = {'zero_shot': 'Zero-shot | pretrained epoch 11',
              'adapter_only': 'Adapter-only | pretrained epoch 11 + shared adapter',
              'joint_epoch29': 'Joint fine-tuning | epoch 29'}
    cross = 'cross_celltype' in set(table.condition)
    conditions = ['correct','matched_cell','cross_celltype','within_cell_gene'] if cross else CONDITIONS
    labels = ['Correct pair', 'Same gene,\nsame-type\ncell S', 'Same gene,\ndifferent-type\ncell S', 'Same cell,\nother gene S'] if cross else ['Correct pair', 'Same gene,\nother cell S', 'Same cell,\nother gene S']
    colors = ['#7498CB', '#E3A356', '#7DAF9A', '#DF7E86'] if cross else ['#7498CB', '#E3A356', '#DF7E86']
    all_values = [q.mean_distance.to_numpy() for _,q in table.groupby(['model','condition'])]
    low = min(np.quantile(v, .1) for v in all_values)
    high = max(np.quantile(v, .9) for v in all_values)
    # Independent figures use local ranges; overview preserves the common scale.
    def draw(ax, model, common_scale=True):
        f = table[table.model == model]
        data = [f[f.condition == c].mean_distance.to_numpy() for c in conditions]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, whis=(10, 90), showfliers=False,
                        medianprops=dict(color='#263747', linewidth=1.8), widths=.55)
        for box, color in zip(bp['boxes'], colors):
            box.set(facecolor=color, alpha=.85)
        if common_scale:
            ax.set_ylim(max(0, low-.04), min(2, high+.04))
        else:
            lo=min(np.quantile(v,.1) for v in data)
            hi=max(np.quantile(v,.9) for v in data)
            margin=max((hi-lo)*.18,1e-6)
            ax.set_ylim(max(0,lo-margin),min(2,hi+margin))
        ax.grid(axis='y', alpha=.2)
        ax.set_title(titles[model], fontsize=14)
    note = ('Forebrain | 1,720 identical cells | all donor-common paired genes; identical gene sets across models and conditions\n'
            'Full-token encoding; pairing changed after encoding. Boxes: 25–75%; whiskers: 10–90%. Descriptive, not an unseen-donor test.')
    if cross:
        note += '\nGene set = target ∩ same-type donor ∩ different-type donor; same-type donor unchanged.'
    def save(fig, name, extra=''):
        fig.text(.5, .017, note+extra, ha='center', fontsize=8.5)
        fig.tight_layout(rect=[0, .17 if cross else .12, 1, .96])
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    for j, model in enumerate(MODELS):
        fig, ax = plt.subplots(figsize=(11 if cross else 9, 7))
        draw(ax, model, common_scale=False)
        ax.set_ylabel('Per-cell mean U–S cosine distance')
        save(fig, f'{j+1:02d}_{model}_pairing', '\nLocal y-axis range; use Figure 04 for a common-scale magnitude comparison.')
    fig, axs = plt.subplots(1, 3, figsize=(23 if cross else 18, 7 if cross else 6.5), sharey=True)
    for ax, model in zip(axs, MODELS):
        draw(ax, model)
    axs[0].set_ylabel('Per-cell mean U–S cosine distance')
    save(fig, '04_three_models_common_scale')
    full = table[table.condition == 'correct'][['model','cell_id','mean_distance']].rename(columns={'mean_distance':'correct_distance'})
    paired = table.merge(full, on=['model','cell_id'], validate='many_to_one')
    paired['delta_distance'] = paired.mean_distance - paired.correct_distance
    paired.to_csv(out / 'paired_changes.csv', index=False)
    fig, axs = plt.subplots(1, 3, figsize=(23 if cross else 18, 7 if cross else 6.5), sharey=True)
    for ax, model in zip(axs, MODELS):
        f = paired[paired.model == model]
        data = [f[f.condition == c].delta_distance for c in conditions[1:]]
        bp = ax.boxplot(data, tick_labels=labels[1:], patch_artist=True, whis=(10,90), showfliers=False)
        for box, color in zip(bp['boxes'], colors[1:]):
            box.set(facecolor=color, alpha=.85)
        ax.axhline(0, color='#555', lw=1, ls='--')
        ax.set_title(titles[model], fontsize=12)
        ax.grid(axis='y', alpha=.2)
    axs[0].set_ylabel('Per-cell Δdistance = null − correct')
    save(fig, '05_three_models_paired_changes')
    summary = paired.groupby(['model','condition'],as_index=False).agg(cells=('cell_id','nunique'),
        mean_distance=('mean_distance','mean'), mean_delta=('delta_distance','mean'),
        median_delta=('delta_distance','median'), fraction_null_gt_correct=('delta_distance',lambda x:(x>0).mean()))
    summary.to_csv(out / 'summary.csv', index=False)
    if cross:
        wide=table.pivot(index=['model','cell_id'],columns='condition',values='mean_distance').reset_index()
        wide['cross_minus_same_type']=wide.cross_celltype-wide.matched_cell
        wide.to_csv(out/'cross_vs_same_type_cells.csv',index=False)
        wide.groupby('model',as_index=False).agg(cells=('cell_id','nunique'),
            mean_cross_minus_same=('cross_minus_same_type','mean'),
            median_cross_minus_same=('cross_minus_same_type','median'),
            fraction_cross_gt_same=('cross_minus_same_type',lambda x:(x>0).mean())).to_csv(out/'cross_vs_same_type_summary.csv',index=False)
    paired.groupby(['model','condition','sample','cell_type'],as_index=False).agg(cells=('cell_id','nunique'),
        mean_delta=('delta_distance','mean'), fraction_null_gt_correct=('delta_distance',lambda x:(x>0).mean())).to_csv(out/'stratified_summary.csv',index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--cross-celltype', action='store_true')
    args = p.parse_args()
    selftest()
    if args.selftest:
        print('Selftest passed'); return
    cfg = load_config(args.config)
    root = Path(cfg['output_dir']) / cfg['tag']
    out = Path('outputs/paper/pretraining_ablations/forebrain_align_pairing') / ('three_models_cross_celltype' if args.cross_celltype else 'three_models_fullinput')
    conditions = ['correct','matched_cell','cross_celltype','within_cell_gene'] if args.cross_celltype else CONDITIONS
    out.mkdir(exist_ok=True)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status') == 'complete':
        print('Already complete'); return
    lock = out/'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        write_json(out/'status.json', dict(status='preflight', pid=os.getpid()))
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('Physical GPU2 required')
        from scus.ablations import resource_preflight
        ok, trace, reason = resource_preflight(2, 31, 10, minimum_free_mb=61440, maximum_mean_utilization=10)
        trace.to_csv(out/'preflight.csv', index=False)
        if not ok:
            raise RuntimeError(reason)
        device = torch.device('cuda:2')
        props = torch.cuda.get_device_properties(device)
        if 'A100' not in props.name or props.total_memory < 75*2**30:
            raise ValueError('80GB A100 required')
        torch.cuda.set_per_process_memory_fraction(60*2**30/props.total_memory, device)
        data = load_joint_data(cfg)
        meta = pd.read_csv(root/'align-fit/cell_manifest.csv').sort_values('cell_index').reset_index(drop=True)
        assert list(meta.cell_id) == list(data.cells) and len(meta) == 1720
        meta['cell_type'] = np.asarray(data.shard['obs_clusters'], dtype=str)
        meta['sample'] = meta.cell_id.str.split(':').str[0]
        genes, valid_positions = [], []
        for i in range(len(meta)):
            b = data[i]; digest = hashlib.sha256()
            for key in ['gene_ids','value_bins','splice_flags']:
                digest.update(b[key].numpy().tobytes())
            if digest.hexdigest() != meta.iloc[i].input_sha256:
                raise ValueError('Training input mismatch')
            v = b['value_bins'][0].numpy(); g = b['gene_ids'][0,::2].numpy()
            pos = np.flatnonzero((v[::2] > 0) & (v[1::2] > 0))
            assert len(np.unique(g[pos])) == len(pos)
            genes.append(g[pos]); valid_positions.append(pos)
        pairs, manifest = build_cross_pairs(meta, genes) if args.cross_celltype else build_pairs(meta, genes)
        manifest.to_csv(out/'cell_pair_manifest.csv', index=False)
        ptr = np.r_[0, np.cumsum([len(g) for g in genes])]
        selection = json.loads((root/'align-fit/loss_selection.json').read_text())
        assert selection['selected_epoch'] == 29
        joint = root/'align-fit/epoch_029.pt'
        old = json.loads(Path('outputs/paper/pretraining_ablations/forebrain_align_pairing/status.json').read_text())
        adapter_path = Path(old['projector']['path'])
        identity = dict(token_sha256=sha256(cfg['data']['token_path']), vocab_sha256=sha256(cfg['vocab']),
                        pretrained_sha256=sha256(cfg['checkpoint']), joint_sha256=sha256(joint),
                        adapter_sha256=sha256(adapter_path), script_sha256=sha256(Path(__file__)),
                        gene_policy='all donor-common valid paired genes, no 512 cap', seed=42,
                        input='same original full tokens; single-modality valid genes kept as attention context',
                        donor='same cell type, nonself, cross-sample preferred', joint_epoch=29)
        if args.cross_celltype:
            previous=pd.read_csv(out.parent/'three_models_fullinput/cell_pair_manifest.csv')
            assert np.array_equal(manifest.donor_cell_id.to_numpy(),previous.donor_cell_id.to_numpy())
            identity.update(gene_policy='all triple-intersection valid paired genes, no cap',
                conditions=conditions,cross_donor='different cell type; same sample as same-type donor preferred; independent per-cell seed',
                original_same_type_manifest_sha256=sha256(out.parent/'three_models_fullinput/cell_pair_manifest.csv'))
        if (out/'run_config.json').exists() and json.loads((out/'run_config.json').read_text()) != identity:
            raise ValueError('Resume provenance mismatch')
        write_json(out/'run_config.json', identity)
        resources = []
        for stage in ['zero_shot', 'joint_epoch29']:
            required = ['zero_shot', 'adapter_only'] if stage == 'zero_shot' else ['joint_epoch29']
            if all((out/f'{name}_cells.csv').exists() for name in required):
                continue
            model = JointModel.from_checkpoint(cfg['checkpoint']).to(device).eval()
            model.requires_grad_(False)
            if stage == 'joint_epoch29':
                payload = torch.load(joint, map_location='cpu', weights_only=False)
                assert payload['identity']['token_source_sha256'] == identity['token_sha256']
                assert payload['identity']['checkpoint_sha256'] == identity['pretrained_sha256']
                model.load_state_dict(payload['model'], strict=True); del payload
            hu = np.empty((int(ptr[-1]),128), dtype=np.float32)
            hs = np.empty_like(hu)
            begin = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
            # ~10.4 GiB host RAM; discarded after this stage, no large embedding files.
            for i,(b,h,z) in enumerate(encoded_cells(model,data,list(range(len(meta))),device,32)):
                value = h if stage == 'zero_shot' else z
                positions = torch.as_tensor(valid_positions[i], device=device)
                hu[ptr[i]:ptr[i+1]] = value[0,2*positions].float().cpu().numpy()
                hs[ptr[i]:ptr[i+1]] = value[0,2*positions+1].float().cpu().numpy()
                if (i+1)%128 == 0 or i+1 == len(meta):
                    state = dict(status='encoding',stage=stage,cells_completed=i+1,total=len(meta),pid=os.getpid(),seconds=time.monotonic()-begin)
                    write_json(out/'status.json',state); print(state,flush=True)
            del model,b,h,z,value; torch.cuda.empty_cache()
            if not np.isfinite(hu).all() or not np.isfinite(hs).all():
                raise ValueError('Nonfinite embedding')
            adapter = None
            if stage == 'zero_shot':
                adapter,_ = load_projector(adapter_path,128); adapter.to(device).eval(); adapter.requires_grad_(False)
            records = {name:[] for name in required}
            for i,item in enumerate(pairs):
                if args.cross_celltype:
                    donor,cross,own,other,cross_pos,permutation=item
                else:
                    donor,own,other,permutation=item
                u = hu[ptr[i]+own]; s = hs[ptr[i]+own]; donor_s = hs[ptr[donor]+other]
                arrays=(u,s,donor_s,hs[ptr[cross]+cross_pos]) if args.cross_celltype else (u,s,donor_s)
                blocks = {stage:arrays}
                if adapter is not None:
                    with torch.inference_mode():
                        combined = adapter(torch.as_tensor(np.concatenate(arrays),device=device)).float().cpu().numpy()
                    blocks['adapter_only'] = np.split(combined,len(arrays))
                for name,embeddings in blocks.items():
                    a,b,d=embeddings[:3]
                    partners=[b,d,embeddings[3],b[permutation]] if args.cross_celltype else [b,d,b[permutation]]
                    for condition,partner in zip(conditions,partners):
                        distance = cosine(a,partner)
                        if not np.isfinite(distance).all() or distance.min() < -1e-6 or distance.max() > 2+1e-6:
                            raise ValueError('Invalid distance')
                        row = manifest.iloc[i].to_dict()
                        row.update(model=name,condition=condition,mean_distance=float(distance.mean()),median_gene_distance=float(np.median(distance)))
                        records[name].append(row)
                if (i+1)%256 == 0:
                    state=dict(status='pairing',stage=stage,cells_completed=i+1,pid=os.getpid())
                    write_json(out/'status.json',state); print(state,flush=True)
            for name in required:
                f = pd.DataFrame(records[name]); tmp=out/f'{name}_cells.partial.csv'
                f.to_csv(tmp,index=False); os.replace(tmp,out/f'{name}_cells.csv')
            resources.append(dict(stage=stage,seconds=time.monotonic()-begin,paired_tokens=int(ptr[-1]),
                                  peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30))
            pd.DataFrame(resources).to_csv(out/'resources.csv',index=False)
            del hu,hs,adapter; torch.cuda.empty_cache()
        table = pd.concat([pd.read_csv(out/f'{name}_cells.csv') for name in MODELS],ignore_index=True)
        assert len(table) == 1720*3*len(conditions) and not table.duplicated(['model','cell_id','condition']).any()
        assert table.groupby('cell_id').n_pairs.nunique().eq(1).all()
        assert table.groupby(['model','condition']).cell_id.nunique().eq(1720).all()
        table.to_csv(out/'pairing_long.csv',index=False)
        write_json(out/'status.json',dict(status='plotting',pid=os.getpid()))
        plots(out,table)
        (out/'README_CN.md').write_text('''# 相同完整输入下的三模型配对消融

三部分单独绘图：Zero-shot（预训练epoch11的h）、adapter-only（相同h经旧共享projector所得z）、联合微调（当前已选epoch29完整模型的z）。不重新训练，不改变checkpoint选择。

所有模型使用相同Forebrain_all_gene.pt完整输入、原始基因顺序与bins，全部1720细胞。单模态有效基因仍参与attention；distance仅在细胞与同cell-type非self donor共同的双模态有效基因上计算，使用全部共同基因，不限512，不是全转录组承诺。cell_pair_manifest记录donor、共同基因数及索引/置换哈希。

Correct：本细胞同gene的U/S；Matched cell：本细胞U与donor同gene的S；Within-cell gene：本细胞U与本细胞其他gene的S。后者用seed42+cell ID派生的随机循环错排，保证无gene self-match。所有条件共用目标集合。这是编码后配对检验，不是Context shuffle，不重新编码被打乱的token输入，也没有24基因面板。

GPU2，BF16 batch32编码；cosine与旧adapter投影使用FP32。编码数组只在主机内存临时保存，不产生大embedding文件。完整模型阶段完成后保存CSV，可恢复已完成阶段；中断的编码阶段需要重算。

01/02/03为独立模型绝对距离图，分别采用明确标注的局部纵轴范围，以显示很小的adapter-only距离；04为三图同尺度总览，用于比较绝对量级；05为各细胞null减correct的配对变化，共用纵轴。箱体25–75%，须线10–90%，每个观测是一细胞的全部共同基因平均距离。不是置信区间。正变化比例不是分类accuracy。

旧adapter训练见过全部Forebrain细胞，联合微调按原split训练；本次全体细胞仅描述性评估，不宣称三者都是unseen-donor或zero-shot协议。旧8,000基因缓存/512配对图不再直接拼接，本轮Zero-shot和adapter-only也重新编码。Matched donor按cell type与sample选择，没有表达量/检测率精确匹配。不同cell可有不同共同基因集合；同一cell的三模型与三条件集合一致。
''')
        if args.cross_celltype:
            with (out/'README_CN.md').open('a') as handle:
                handle.write('''

## 新增不同cell-type条件（本目录的实际四条件协议）

Same gene, different-type cell S：目标U不变，使用不同cell type donor的同gene S。优先从原同类型donor的sample选择，避免两种donor来源sample系统不同；随机seed由42和cell ID派生。原同类型donor逐细胞保持不变。

本目录覆盖上文的两方交集描述：四条件实际共用目标细胞、同类型donor及不同类型donor的三方共同有效配对基因。所有条件重新汇总，故旧三组的均值可以变化；不能将这里的新增组直接拼到旧两方交集图中。编码输入仍保留全部来源有效基因，无基因截断。本次保存三方交集的索引哈希及两类donor字段。

新绿色箱体为不同类型donor，其余颜色延续。cross_vs_same_type_summary.csv直接比较每细胞不同类型与同类型donor的距离；不是仅比较两组无配对总体均值。标签仅用于构造描述性null，不能将其当作无标签异常检测性能，也未表达/检测率匹配。
''')
        write_json(out/'status.json',dict(status='complete',cells=1720,models=3,conditions=len(conditions),rows=len(table),joint_epoch=29,no_retraining=True))
        print('COMPLETE',out,flush=True)
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',pid=os.getpid(),error=str(exc))); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
