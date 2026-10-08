from pathlib import Path
import os,json,hashlib,time,fcntl,subprocess,contextlib
ROOT=Path(__file__).resolve().parents[1]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ.setdefault(k,'4')
for k,v in [('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda'),('TORCHINDUCTOR_CACHE_DIR','torchinductor')]:
 p=ROOT/'.cache'/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import numpy as np
import pandas as pd

def read_json(p):return json.loads(Path(p).read_text())
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=Path(str(p)+'.'+str(os.getpid())+'.partial');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def protocol():return read_json(ROOT/'configs/protocol.json')
def specs():return read_json(ROOT/'configs/datasets.json')
def folder(name,smoke=False):return ROOT/('tests/smoke' if smoke else 'datasets')/name
def status(out,stage,state,**kw):
 rec={'stage':stage,'status':state,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),**kw};dump(out/'status.json',rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
def signature(name):return hashlib.sha256(json.dumps({'protocol':protocol(),'dataset':specs()[name]},sort_keys=True).encode()).hexdigest()
def done(out,stage,name):
 p=out/f'{stage}.complete.json'
 if not p.exists():return False
 r=read_json(p)
 if r['signature']!=signature(name):raise ValueError(f'Configuration changed: {p}')
 for rel,h in r['outputs'].items():
  if not (out/rel).exists() or sha(out/rel)!=h:raise ValueError(f'Changed/missing output: {out/rel}')
 return True
def complete(out,stage,name,paths,**kw):
 dump(out/f'{stage}.complete.json',dict(signature=signature(name),outputs={str(Path(p).relative_to(out)):sha(p) for p in paths},code={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},**kw))
@contextlib.contextmanager
def lock(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError(f'Active lock: {path}')
  f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush()
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
@contextlib.contextmanager
def dataset_lock(path, wait=False):
 # Only the all-data coordinator waits; direct duplicate dataset starts still fail.
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:
   if not wait:raise RuntimeError(f'Active lock: {path}')
   print(f'Waiting for independently running dataset: {path}',flush=True)
   fcntl.flock(f,fcntl.LOCK_EX)
  f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush()
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)

@contextlib.contextmanager
def gpu(wait=True):
 while True:
  result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  if result.returncode:raise RuntimeError('GPU driver visibility check failed: '+result.stderr+result.stdout)
  records=[tuple(map(int,r.split(','))) for r in result.stdout.strip().splitlines()]
  selected=None
  for idx,total,free,use in sorted(records,key=lambda r:-r[2]):
   if free<min(61440,int(total*.85))+2048 or use>=10:continue
   f=(ROOT/'locks'/f'gpu{idx}.lock').open('a+')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:f.close();continue
   selected=(idx,total,f);break
  if selected:break
  if not wait:raise RuntimeError('No idle GPU passes resource preflight')
  print('Waiting for idle GPU; retry in 30 seconds',flush=True);time.sleep(30)
 idx,total,f=selected;old=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=str(idx)
 try:yield idx,total
 finally:
  if old is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=old
  fcntl.flock(f,fcntl.LOCK_UN);f.close()

def run_child(cmd,log,needs_gpu=False):
 log.parent.mkdir(parents=True,exist_ok=True)
 def execute(device=None):
  with log.open('a') as stream:
   stream.write(f'\nCOMMAND {cmd!r} physical_gpu={device}\n');stream.flush()
   r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=os.environ.copy())
  if r.returncode:raise RuntimeError(f'Child failed ({r.returncode}): {log}')
 if needs_gpu:
  with gpu() as (idx,total):execute(idx)
 else:execute()

def verify_external_inputs(name):
 audit=read_json(folder(name)/'data/audit.json')
 for path,expected in audit['input_hashes'].items():
  if sha(path)!=expected:raise ValueError('External input changed: '+path)
 weights=read_json(ROOT/'provenance/model_hashes.json')
 for path,rec in weights.items():
  if sha(path)!=rec['sha256']:raise ValueError('Model/vocabulary changed: '+path)
 return dict(dataset=name,verified_inputs=len(audit['input_hashes']),verified_model_files=len(weights),time=time.strftime('%Y-%m-%d %H:%M:%S'))
