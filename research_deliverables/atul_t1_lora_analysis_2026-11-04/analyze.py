"""CPU-only read-only safetensors analysis. Never imports downloaded code or writes tensors.
Run with --integrity-only first, then without it. Outputs require a fresh destination.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse, collections, csv, hashlib, json, math, pathlib, re, struct, sys, time
import numpy as np
ROOT=pathlib.Path('/home/atul-baliyan/Desktop/gemma4-t1-weights')
EXPECTED={'step25':'3804c9260680c1c110372d3b3fa75f2c9b27e8ca6319fd9c6daf4658e10d751b','final':'2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510','serving':'2b22c3f393f178f0c8885df99229a501da826ec1150bda83d8bfd7f4d3d56510'}
PATHS={'step25':'train_out/adapter_step25','final':'train_out/adapter_final','serving':'adapters/rs2_lora'}
PREFIX='base_model.model.model.language_model.layers.'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('Duplicate JSON key '+k)
  d[k]=v
 return d

def expected_shapes():
 h={}
 for i in range(60):
  glob=i%6==5
  dims={'self_attn.q_proj':(16384 if glob else 8192,5376),'self_attn.k_proj':(2048 if glob else 4096,5376),'self_attn.o_proj':(5376,16384 if glob else 8192),'mlp.gate_proj':(21504,5376),'mlp.up_proj':(21504,5376),'mlp.down_proj':(5376,21504)}
  if not glob:dims['self_attn.v_proj']=(4096,5376)
  for name,(out,inp) in dims.items():
   key=PREFIX+str(i)+'.'+name
   h[key+'.lora_A.weight']=[16,inp];h[key+'.lora_B.weight']=[out,16]
 return h

class Checkpoint:
 def __init__(self,label):
  p=ROOT/PATHS[label]/'adapter_model.safetensors';self.path=p
  digest=sha(p);assert digest==EXPECTED[label],(label,'SHA256 mismatch')
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];assert 0<n<10_000_000 and n%8==0
   self.h=json.loads(f.read(n),object_pairs_hook=unique)
  metadata=self.h.pop('__metadata__',{})
  assert isinstance(metadata,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in metadata.items())
  exp=expected_shapes();assert self.h.keys()==exp.keys(),(label,'key set mismatch')
  cursor=0;count=0
  for k,v in sorted(self.h.items(),key=lambda x:x[1]['data_offsets'][0]):
   assert v['shape']==exp[k] and v['dtype']=='BF16',(label,k)
   a,b=v['data_offsets'];assert isinstance(a,int) and isinstance(b,int) and a==cursor
   assert b-a==math.prod(v['shape'])*2 and b<=p.stat().st_size-8-n
   cursor=b;count+=math.prod(v['shape'])
  assert cursor==p.stat().st_size-8-n and count==122429440
  cfgp=p.parent/'adapter_config.json';cfg=json.loads(cfgp.read_text(),object_pairs_hook=unique)
  assert cfg['r']==16 and cfg['lora_alpha']==32 and cfg['lora_dropout']==0 and cfg['bias']=='none'
  assert cfg['fan_in_fan_out'] is False and cfg['peft_type']=='LORA' and cfg['inference_mode'] is True
  assert set(cfg['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
  assert not any(cfg.get(k) for k in ['rank_pattern','alpha_pattern','use_rslora','use_dora','modules_to_save'])
  self.cfg=cfg;self.mm=np.memmap(p,mode='r',dtype=np.uint8,offset=8+n)
  zeros=0
  for k in self.h:
   x=self.tensor(k);assert np.isfinite(x).all(),(label,k,'nonfinite');zeros+=int(np.count_nonzero(x==0))
  self.info={'path':str(p),'sha256':digest,'bytes':p.stat().st_size,'header_bytes':n,'tensors':len(self.h),'modules':410,'parameters':count,'dtype':'BF16','zero_factor_entries':zeros,'finite':True,'contiguous_payload':True,'config':cfg,'config_sha256':sha(cfgp),'scale':2}
 def tensor(self,k):
  v=self.h[k];a,b=v['data_offsets'];u=np.frombuffer(self.mm[a:b],dtype='<u2').astype('<u4')<<16
  return u.view('<f4').reshape(v['shape']).astype(np.float64)
 def factors(self,m):return self.tensor(m+'.lora_A.weight'),self.tensor(m+'.lora_B.weight')

def spectrum(a,b):
 rb=np.linalg.qr(b,mode='r');ra=np.linalg.qr(a.T,mode='r')
 return np.linalg.svd(2*rb@ra.T,compute_uv=False)

def stats(sv,out,inp):
 e=sv**2;energy=float(e.sum());p=e/energy if energy else e
 return {'energy':energy,'frobenius':math.sqrt(energy),'spectral':float(sv[0]),'stable_rank':energy/float(e[0]) if energy else 0,'effective_rank_energy':float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))) if energy else 0,'update_rms':math.sqrt(energy/(out*inp)),'top1_energy':float(p[:1].sum()),'top4_energy':float(p[:4].sum()),'top8_energy':float(p[:8].sum()),'rank90':int(np.searchsorted(np.cumsum(p),.9)+1) if energy else 0,'rank99':int(np.searchsorted(np.cumsum(p),.99)+1) if energy else 0}

def gram(a,b):return float(4*np.sum((b.T@b)*(a@a.T).T))
def cross(a,b,c,d):return float(4*np.sum((b.T@d)*(a@c.T)))
def dump(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def csvout(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--integrity-only',action='store_true');ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args();args.out.mkdir(exist_ok=True)
 t=time.time();ck={k:Checkpoint(k) for k in PATHS}
 assert ck['step25'].cfg==ck['final'].cfg==ck['serving'].cfg
 # Explicit stream byte equality, independent of hash comparison.
 with ck['final'].path.open('rb') as a,ck['serving'].path.open('rb') as b:
  while True:
   x,y=a.read(1<<20),b.read(1<<20);assert x==y
   if not x:break
 integ={k:v.info for k,v in ck.items()};integ['final_serving_byte_identical']=True
 print('INTEGRITY PASS: all three, 820 tensors each, configs equal, final/serving identical',flush=True)
 if args.integrity_only:dump(args.out/'integrity.json',integ);return
 validation={'numpy':np.__version__,'python':sys.version,'cpu_only':True,'synthetic_max_relative_error':0,'gram_max_relative_error':0,'difference_gram_max_relative_error':0,'submatrix_max_absolute_error':0}
 rng=np.random.default_rng(20261104)
 for mode in ('random','rank_deficient','zero'):
  a=rng.normal(size=(4,17));b=rng.normal(size=(23,4))
  if mode=='rank_deficient':b[:,2:]=0
  if mode=='zero':b[:]=0
  small=2*b@a;s=spectrum(a,b);d=np.linalg.svd(small,compute_uv=False)[:4]
  err=float(np.linalg.norm(s-d)/max(1,np.linalg.norm(d)));assert err<1e-12
  validation['synthetic_max_relative_error']=max(validation['synthetic_max_relative_error'],err)
  c=rng.normal(size=a.shape);e=rng.normal(size=b.shape)
  sd=spectrum(np.vstack((a,-c)),np.hstack((b,e)))
  dense=2*(b@a-e@c);assert np.allclose(sd,np.linalg.svd(dense,compute_uv=False)[:8],rtol=1e-12,atol=1e-12)
  assert np.isclose(cross(a,b,c,e),float(np.sum(small*(2*e@c))),rtol=1e-12,atol=1e-12)
 rows=[];comparisons=[];spectra=[]
 modules=sorted(k.removesuffix('.lora_A.weight') for k in ck['final'].h if k.endswith('.lora_A.weight'))
 for j,m in enumerate(modules):
  i=int(m.split('.layers.')[1].split('.')[0]);family=m.split('.')[-1];comp='attention' if '.self_attn.' in m else 'mlp';typ='global' if i%6==5 else 'sliding'
  a,b=ck['step25'].factors(m);c,d=ck['final'].factors(m);out,inp=b.shape[0],a.shape[1]
  info={'module':m,'layer':i,'family':family,'component':comp,'attention_type':typ,'depth12':f'{i//12*12:02d}-{i//12*12+11:02d}','dense_elements':out*inp,'lora_parameters':a.size+b.size}
  result=[]
  for name,x,y in [('step25',a,b),('final',c,d)]:
   sv=spectrum(x,y);st=stats(sv,out,inp);result.append(st)
   err=abs(st['energy']-gram(x,y))/max(st['energy'],1e-300);assert err<1e-10
   validation['gram_max_relative_error']=max(validation['gram_max_relative_error'],err)
   rows.append(dict(checkpoint=name,**info,**st,base_relative_rms='BLOCKED_missing_base_weights'))
   spectra.append({'checkpoint':name,'module':m,'singular_values':sv.tolist(),'energy_fractions':(sv**2/st['energy']).tolist()})
  s0,s1=result;inner=cross(a,b,c,d);diffsv=spectrum(np.vstack((c,-a)),np.hstack((d,b)));diff=stats(diffsv,out,inp)
  err=abs(diff['energy']-(s0['energy']+s1['energy']-2*inner))/max(diff['energy'],1e-300);assert err<1e-9
  validation['difference_gram_max_relative_error']=max(validation['difference_gram_max_relative_error'],err)
  # Real-weight tiny rectangular slice: direct multiplication vs rank-one accumulation.
  if j%20==0:
   dense=2*(d[:7]@c[:,:9]-b[:7]@a[:,:9]);ind=sum(2*(np.outer(d[:7,k],c[k,:9])-np.outer(b[:7,k],a[k,:9])) for k in range(16))
   e=float(np.max(np.abs(dense-ind)));assert e<1e-12;validation['submatrix_max_absolute_error']=max(validation['submatrix_max_absolute_error'],e)
  comparisons.append(dict(**info,step25_frobenius=s0['frobenius'],final_frobenius=s1['frobenius'],norm_growth=s1['frobenius']/s0['frobenius'],cosine=inner/(s0['frobenius']*s1['frobenius']),difference_frobenius=diff['frobenius'],relative_change_vs_step25=diff['frobenius']/s0['frobenius'],difference_energy=diff['energy'],difference_update_rms=diff['update_rms'],difference_spectral=diff['spectral'],difference_stable_rank=diff['stable_rank'],difference_effective_rank_energy=diff['effective_rank_energy'],inner_product=inner))
  if j%100==0:print('Analyzed',j+1,'/410',flush=True)
 assert len(rows)==820 and len(comparisons)==410
 groups=[]
 for group in ['all','family','component','layer','depth12','attention_type','family_type']:
  for name in ['step25','final']:
   sets=collections.defaultdict(list)
   for r in rows:
    if r['checkpoint']!=name:continue
    key='all' if group=='all' else (r['family']+':'+r['attention_type'] if group=='family_type' else str(r[group]))
    sets[key].append(r)
   for key,rs in sets.items():
    es=sum(r['energy'] for r in rs);cs=[c for c in comparisons if c['module'] in {r['module'] for r in rs}]
    groups.append({'checkpoint':name,'group':group,'key':key,'modules':len(rs),'energy':es,'frobenius_root_sum_squares':math.sqrt(es),'pooled_update_rms':math.sqrt(es/sum(r['dense_elements'] for r in rs)),'median_update_rms':float(np.median([r['update_rms'] for r in rs])),'median_stable_rank':float(np.median([r['stable_rank'] for r in rs])),'median_effective_rank_energy':float(np.median([r['effective_rank_energy'] for r in rs])),'median_top1_energy':float(np.median([r['top1_energy'] for r in rs])),'difference_energy':sum(c['difference_energy'] for c in cs),'median_cosine':float(np.median([c['cosine'] for c in cs])),'median_relative_change':float(np.median([c['relative_change_vs_step25'] for c in cs]))})
 csvout(args.out/'modules.csv',rows);csvout(args.out/'checkpoint_comparison.csv',comparisons);csvout(args.out/'groups.csv',groups)
 dump(args.out/'spectra.json',spectra)
 validation.update(elapsed_seconds=time.time()-t,all_410_comparisons_pass=True,real_submatrix_checks=21,synthetic_cases=['random','rank_deficient','zero'],script_sha256=sha(pathlib.Path(__file__)))
 dump(args.out/'validation.json',validation)
 print(json.dumps(validation),flush=True)
if __name__=='__main__':main()
