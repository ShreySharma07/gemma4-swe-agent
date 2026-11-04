"""Derived numerical summaries only; no tensor serialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,math,pathlib,sys
import numpy as np
p=pathlib.Path(__file__).parent
rows=list(csv.DictReader((p/'modules.csv').open()));cs=list(csv.DictReader((p/'checkpoint_comparison.csv').open()));groups=list(csv.DictReader((p/'groups.csv').open()))
for rows_ in [rows,cs,groups]:
 for r in rows_:
  for k,v in r.items():
   try:r[k]=float(v)
   except ValueError:pass
f=[r for r in rows if r['checkpoint']=='final'];s=[r for r in rows if r['checkpoint']=='step25']
e0=sum(r['energy'] for r in s);e1=sum(r['energy'] for r in f);de=sum(r['difference_energy'] for r in cs);inner=sum(r['inner_product'] for r in cs)
out={'step25_energy':e0,'final_energy':e1,'energy_growth_pct':100*(e1/e0-1),'frobenius_step25':math.sqrt(e0),'frobenius_final':math.sqrt(e1),'frobenius_growth_pct':100*(math.sqrt(e1/e0)-1),'difference_frobenius':math.sqrt(de),'relative_change_global':math.sqrt(de/e0),'concatenated_cosine':inner/math.sqrt(e0*e1),'all_modules_norm_increased':all(r['norm_growth']>1 for r in cs),'cosine_min_max':[min(r['cosine'] for r in cs),max(r['cosine'] for r in cs)],'delta_relative_min_max':[min(r['relative_change_vs_step25'] for r in cs),max(r['relative_change_vs_step25'] for r in cs)]}
for key in ['stable_rank','effective_rank_energy','top1_energy','top4_energy','top8_energy','rank90','rank99']:
 out['final_'+key+'_quantiles']=np.quantile([r[key] for r in f],[0,.25,.5,.75,1]).tolist()
for key in ['family','component','depth12']:
 out[key]=[]
 for g in groups:
  if g['checkpoint']=='final' and g['group']==key:
   g=dict(g);g['energy_share_pct']=100*g['energy']/e1;g['change_energy_share_pct']=100*g['difference_energy']/de
   out[key].append(g)
out['attention_by_type']=[]
for typ in ['global','sliding']:
 rr=[r for r in f if r['component']=='attention' and r['attention_type']==typ]
 out['attention_by_type'].append({'type':typ,'n':len(rr),'energy':sum(r['energy'] for r in rr),'pooled_rms':math.sqrt(sum(r['energy'] for r in rr)/sum(r['dense_elements'] for r in rr)),'median_stable_rank':float(np.median([r['stable_rank'] for r in rr]))})
out['attention_family_type']=[g for g in groups if g['checkpoint']=='final' and g['group']=='family_type' and not g['key'].startswith(('gate','up','down'))]
out['top_layers']=sorted([g for g in groups if g['checkpoint']=='final' and g['group']=='layer'],key=lambda x:x['energy'],reverse=True)[:10]
out['rankings']={key:sorted(f,key=lambda r:r[key],reverse=True)[:10] for key in ['frobenius','update_rms','spectral','effective_rank_energy']}
out['change_rankings']={key:sorted(cs,key=lambda r:r[key],reverse=key!='cosine')[:10] for key in ['difference_frobenius','relative_change_vs_step25','cosine']}
extra=[];cm={r['module']:r for r in cs}
for r in json.loads((p/'spectra.json').read_text()):
 sv=np.array(r['singular_values']);pr=sv/sv.sum();extra.append({'checkpoint':r['checkpoint'],'module':r['module'],'effective_rank_singular_entropy':float(np.exp(-np.sum(pr*np.log(pr)))),'rms_final_over_step25':cm[r['module']]['norm_growth']})
with (p/'supplementary_metrics.csv').open('x',newline='') as f_:
 w=csv.DictWriter(f_,fieldnames=list(extra[0]));w.writeheader();w.writerows(extra)
with (p/'summary.json').open('x') as f_:json.dump(out,f_,indent=2,allow_nan=False)
# Independent streamed numerical sanity checks, using only our authored parser.
import analyze
rng=np.random.default_rng(41);ck=analyze.Checkpoint('final');checks=[]
for module in [analyze.PREFIX+'59.mlp.up_proj',analyze.PREFIX+'11.self_attn.o_proj',analyze.PREFIX+'4.self_attn.k_proj']:
 a,b=ck.factors(module)
 ai=a[:,:31];bi=b[:23,:];dense=2*bi@ai
 expected=np.linalg.svd(dense,compute_uv=False)[:16];actual=analyze.spectrum(ai,bi)
 assert np.allclose(actual,expected,rtol=1e-11,atol=1e-14)
 # Gauge invariance under orthogonal change of factor basis; no checkpoint writes.
 q=np.linalg.qr(rng.normal(size=(16,16)))[0];orig=analyze.spectrum(a,b);rot=analyze.spectrum(q.T@a,b@q)
 error=float(np.max(np.abs(rot-orig))/orig[0]);assert error<1e-11
 checks.append({'module':module,'actual_slice_svd_max_error':float(np.max(np.abs(actual-expected))),'gauge_spectrum_max_relative_error':error})
with (p/'independent_checks.json').open('x') as f_:json.dump(checks,f_,indent=2)
print({k:v for k,v in out.items() if not isinstance(v,(list,dict))})
print('attention',out['attention_by_type'])
print('Independent slice SVD and gauge tests PASS')
