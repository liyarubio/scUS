"""Environment-specific real GPU encoders. Full scUS tokens, atomic per-stage output."""
from common import *
import sys,random

def init_cuda():
 import torch
 if not torch.cuda.is_available():raise RuntimeError('Real CUDA required; CPU fallback is forbidden')
 torch.cuda.set_per_process_memory_fraction(min(.90,60*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
 print('CUDA',torch.__version__,torch.cuda.get_device_name(0),'physical',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
 return torch

def scus(out):
 torch=init_cuda();sys.path.insert(0,str(ROOT/'code/vendor'))
 from scus.joint_reconstruction import JointModel
 from scus.joint_batch import collate
 model=JointModel.from_checkpoint(protocol()['scus_checkpoint']).cuda().eval()
 ub=np.load(out/'data/u_bins.npy',mmap_mode='r');sb=np.load(out/'data/s_bins.npy',mmap_mode='r');gids=pd.read_csv(out/'data/scus_genes.csv').vocab_id.to_numpy();n,g=ub.shape;dim=model.encoder.head_gene.in_features
 dst=out/'encoding';dst.mkdir(exist_ok=True);names={'distance':(n,g),'u_sum':(n,g),'s_sum':(n,g),'pooled_u':(n,dim),'pooled_s':(n,dim)}
 progress=read_json(dst/'progress.json') if (dst/'progress.json').exists() else dict(next_cell=0,signature=signature(DATASET))
 assert progress['signature']==signature(DATASET)
 arrays={k:np.load(dst/(k+'.npy'),mmap_mode='r+') if progress['next_cell']==n and (dst/(k+'.npy')).exists() else np.lib.format.open_memmap(dst/(k+'.partial.npy'),mode='r+' if (dst/(k+'.partial.npy')).exists() else 'w+',dtype='float32',shape=shape) for k,shape in names.items()}
 panel=((ub>0)&(sb>0)).sum(0)>=min(protocol()['min_gene_observations'],n)
 for i in range(progress['next_cell'],n):
  keep=np.flatnonzero((ub[i]>0)|(sb[i]>0));values=np.empty(2*len(keep),np.int64);values[::2]=ub[i,keep];values[1::2]=sb[i,keep]
  gg=torch.tensor(np.repeat(gids[keep],2)[None,:],device='cuda');vv=torch.tensor(values[None,:],device='cuda');flags=torch.tensor(np.tile([0,1],len(keep))[None,:],device='cuda')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):h,_=model.hidden(gg,vv,flags)
  u=h[0,::2].float();s=h[0,1::2].float();pair=(ub[i,keep]>0)&(sb[i,keep]>0);pool=pair&panel[keep]
  if not pool.any():raise ValueError(f'No eligible paired genes for scUS pooling, row {i}; cannot silently delete cell')
  for key in ['distance','u_sum','s_sum']:arrays[key][i]=np.nan
  d=(1-torch.nn.functional.cosine_similarity(u,s,dim=-1)).clamp(0,2).cpu().numpy();arrays['distance'][i,keep[pair]]=d[pair]
  arrays['u_sum'][i,keep[ub[i,keep]>0]]=u.sum(-1).cpu().numpy()[ub[i,keep]>0];arrays['s_sum'][i,keep[sb[i,keep]>0]]=s.sum(-1).cpu().numpy()[sb[i,keep]>0]
  arrays['pooled_u'][i]=u[torch.tensor(pool,device='cuda')].double().mean(0).float().cpu().numpy();arrays['pooled_s'][i]=s[torch.tensor(pool,device='cuda')].double().mean(0).float().cpu().numpy()
  for a in arrays.values():a.flush()
  dump(dst/'progress.json',dict(next_cell=i+1,signature=signature(DATASET),last_token_length=len(values)))
  if STOP_AFTER and i+1>=STOP_AFTER:raise SystemExit(99)
  if i%16==0 or i==n-1:print(f'scUS cells={i+1}/{n} tokens={len(values)} memory_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}',flush=True)
 arrays.clear() # close references # references closed below by process; Unix rename safe
 for k in names:
  if (dst/(k+'.partial.npy')).exists():(dst/(k+'.partial.npy')).replace(dst/(k+'.npy'))
 np.save(dst/'pooling_panel.npy',panel)
 dump(dst/'audit.json',dict(model='epoch11 frozen raw hidden before adapter',masked=False,truncated=False,full_input_genes=g,pooling='float64 mean over paired genes with >=50 cohort observations',dtype='bf16 attention / float32 hidden',cells=n,dim=dim,checkpoint_sha256=sha(protocol()['scus_checkpoint'])))

def scgpt(out):
 torch=init_cuda();import scanpy as sc
 sys.path.insert(0,'/data1/liyaru/software/scGPT/scGPT-main')
 from scgpt.tokenizer.gene_tokenizer import GeneVocab
 from scgpt.tasks.cell_emb import get_batch_cell_embeddings
 from scgpt.model import TransformerModel
 from scgpt.utils import load_pretrained
 directory=Path(protocol()['scgpt_model']);vocab=GeneVocab.from_file(str(directory/'vocab.json'));config=read_json(directory/'args.json')
 model=TransformerModel(ntoken=len(vocab),d_model=config['embsize'],nhead=config['nheads'],d_hid=config['d_hid'],nlayers=config['nlayers'],dropout=config['dropout'],vocab=vocab,pad_token='<pad>')
 ckpt=torch.load(directory/'best_model.pt',map_location='cpu');state=ckpt.get('state_dict',ckpt.get('model',ckpt));state={(k[6:] if k.startswith('model.') else k):v for k,v in state.items()};load_pretrained(model,state,strict=False,verbose=False);model.device=torch.device('cuda');model.cuda().eval()
 a=sc.read_h5ad(out/'data/input.h5ad');a=a[:,[g in vocab for g in a.var_names]].copy();a.var['id_in_vocab']=[vocab[g] for g in a.var_names];assert a.n_vars>0
 # Existing official CLS protocol consumes raw counts, with its native 1200-token sampling.
 emb=get_batch_cell_embeddings(a,cell_embedding_mode='cls',model=model,vocab=vocab,batch_size=protocol()['scgpt_batch'],max_length=1200,model_configs={'pad_token':'<pad>','pad_value':-2,'embsize':config['embsize']})
 return emb,dict(genes=a.n_vars,input='raw U+S counts',readout='official CLS max_length1200',checkpoint=sha(directory/'best_model.pt'))

def scvi(out):
 torch=init_cuda();import scvi as sv;import anndata as ad
 sv.settings.seed=42;a=ad.read_h5ad(out/'data/input.h5ad');sv.model.SCVI.setup_anndata(a);model=sv.model.SCVI(a,n_latent=128,n_layers=2,gene_likelihood='zinb')
 model.train(max_epochs=protocol()['scvi_epochs'],accelerator='gpu',devices=1,plan_kwargs={'lr':.01},enable_progress_bar=False)
 dest=out/'models/scVI';dest.mkdir(parents=True,exist_ok=True);model.save(str(dest),overwrite=True);model.history['elbo_train'].to_csv(dest/'training_history.csv')
 return model.get_latent_representation(),dict(input='raw U+S counts',epochs=protocol()['scvi_epochs'],n_latent=128,n_layers=2,lr=.01,target_trained=True)

def foundation(out):
 torch=init_cuda();import scanpy as sc;import omicverse as ov
 from omicverse.llm import SCLLMManager
 a=sc.read_h5ad(folder(DATASET)/'data/input.h5ad');input_genes=a.n_vars;fit_cells=a.n_obs
 a=ov.pp.preprocess(a,mode='shiftlog|pearson',n_HVGs=a.n_vars,batch_key=None,target_sum=1e4)
 mask='highly_variable_features' if 'highly_variable_features' in a.var else 'highly_variable'
 if mask in a.var:a=a[:,a.var[mask]].copy()
 selected_genes=a.var_names.to_numpy().copy();ids=pd.read_csv(out/'data/cells.csv',dtype={'cell_id':str},keep_default_na=False).cell_id.to_numpy();positions=a.obs_names.get_indexer(ids);assert (positions>=0).all();a=a[positions].copy()
 manager=SCLLMManager(model_type='scfoundation',model_path=protocol()['scfoundation_model'],device='cuda',seed=42)
 # Official one-cell pooling excludes padding. Chunk only across cells, never across genes.
 output=[]
 for i in range(0,a.n_obs,32):
  e=manager.get_embeddings(a[i:i+32].copy(),pre_normalized='T',input_type='singlecell');output.append(e);print(f'scFoundation cells={min(i+32,a.n_obs)}/{a.n_obs}',flush=True)
 return np.vstack(output),dict(input='raw U+S -> shiftlog|pearson; cohort-level native gene filtering, no further HVG narrowing',preprocess_fit_cells=fit_cells,input_genes=input_genes,preprocessed_genes=len(selected_genes),gene_list_sha256=hashlib.sha256('\n'.join(selected_genes).encode()).hexdigest(),readout='official last+second-last+max+mean; 3072D',checkpoint=sha(protocol()['scfoundation_model']))

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--model',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--test-output');ap.add_argument('--stop-after',type=int,default=0);args=ap.parse_args();DATASET=args.dataset;STOP_AFTER=args.stop_after;out=folder(DATASET,args.smoke) if not args.test_output else Path(args.test_output);stage='model_'+args.model
 if args.test_output:assert out.resolve().is_relative_to((ROOT/'tests').resolve())
 if done(out,stage,DATASET):sys.exit(0)
 status(out,stage,'running')
 if args.model=='scUS':
  scus(out);paths=list((out/'encoding').glob('*.npy'))+[out/'encoding/audit.json']
 else:
  func={'scGPT':scgpt,'scVI':scvi,'scFoundation':foundation}[args.model];emb,audit=func(out);assert emb.shape[0]==len(pd.read_csv(out/'data/cells.csv')) and np.isfinite(emb).all()
  dst=out/'models'/args.model;dst.mkdir(parents=True,exist_ok=True);np.save(dst/'features.partial.npy',emb.astype(np.float32));(dst/'features.partial.npy').replace(dst/'features.npy');dump(dst/'audit.json',audit);paths=[dst/'features.npy',dst/'audit.json']
 complete(out,stage,DATASET,paths);status(out,stage,'complete')
