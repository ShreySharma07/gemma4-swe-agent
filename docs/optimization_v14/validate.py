"""CPU-only validation of preserved NPZs, evidence identity and paired outcomes."""
import ast
import hashlib
import json
import math
import random
import struct
import tarfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def npy(z,name):
    b=z.read(name+'.npy');assert b[:6]==b'\x93NUMPY'
    offset=10 if b[6]==1 else 12
    n=struct.unpack('<H' if offset==10 else '<I',b[8:offset])[0]
    h=ast.literal_eval(b[offset:offset+n].decode());assert not h['fortran_order']
    return h,b[offset+n:]

def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def main():
    manifest=json.loads((OUT/'input_manifest.json').read_text());changed=[]
    for r in manifest:
        p=ROOT/r['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:changed.append(r['path'])
    assert not changed,changed
    datasets={}
    for name in ['train_public_200','train_s1_own','train_s3_compact','train_s3b']:
        p=ROOT/'data/runs'/(name+'.npz')
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            ih,ib=npy(z,'ids');mh,mask=npy(z,'train');sh,sb=npy(z,'starts');th,tb=npy(z,'task');vh,vb=npy(z,'val')
        assert ih['descr']=='<i4' and mh['descr']=='|u1' and sh['descr']=='<i8'
        starts=list(struct.unpack('<'+'q'*(len(sb)//8),sb));assert starts[0]==0 and starts[-1]==len(mask)==ih['shape'][0]
        assert len(ib)==4*len(mask) and set(mask)<={0,1} and set(vb)<={0,1}
        widths=int(th['descr'].removeprefix('<U'));tasks=[tb[i:i+widths*4].decode('utf-32-le').rstrip('\0') for i in range(0,len(tb),widths*4)]
        assert len(tasks)==len(vb)==len(starts)-1
        lengths=[b-a for a,b in zip(starts,starts[1:])];assert all(0<n<=8192 for n in lengths)
        train_tasks={t for t,v in zip(tasks,vb) if not v};val_tasks={t for t,v in zip(tasks,vb) if v};assert not train_tasks&val_tasks
        labels=[sum(mask[a+1:b]) for a,b in zip(starts,starts[1:])];assert min(labels)>0
        datasets[name]=dict(windows=len(tasks),distinct_tasks=len(set(tasks)),tokens=len(mask),supervised_mask_tokens=sum(mask),shifted_supervised_labels=sum(labels),min_labels=min(labels),max_labels=max(labels),max_length=max(lengths),train_tasks=len(train_tasks),val_tasks=len(val_tasks),train_val_task_overlap=0,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    path=ROOT/'data/runs/atul_eval_v14_new60/baseline_recovery/eval_v7_noedit.tgz'
    with tarfile.open(path) as t:
        members=[m for m in t if m.name.endswith('/merged/task_results.jsonl')];assert len(members)==1
        old=[json.loads(l) for l in t.extractfile(members[0]).read().splitlines()]
    new=json.loads((OUT/'trace_audit.json').read_text())['tasks'];v7={r['instance_id']:r for r in old};v14={r['instance_id']:r for r in new};assert len(v7)==len(v14)==60 and v7.keys()==v14.keys()
    pairs=[(bool(v7[k]['resolved']),bool(v14[k]['resolved'])) for k in sorted(v7)]
    both=sum(a and b for a,b in pairs);gain=sum(not a and b for a,b in pairs);loss=sum(a and not b for a,b in pairs)
    n=gain+loss;pval=min(1.,2*sum(math.comb(n,i) for i in range(min(gain,loss)+1))/2**n)
    rng=random.Random(20261020);diff=[int(b)-int(a) for a,b in pairs];boot=sorted(sum(rng.choices(diff,k=60))/60 for _ in range(20000))
    paired=dict(n=60,v7=sum(a for a,b in pairs),v14=sum(b for a,b in pairs),both=both,gain=gain,loss=loss,neither=60-both-gain-loss,exact_mcnemar_p=pval,v7_wilson95=wilson(sum(a for a,b in pairs),60),v14_wilson95=wilson(sum(b for a,b in pairs),60),paired_task_bootstrap_percentile95=[boot[499],boot[19499]],bootstrap_seed=20261020,bootstrap_replicates=20000,bootstrap_limitation='Task resampling within three public repositories; no stochastic-run or repository-generalization uncertainty.')
    assert (both,gain,loss)==(22,8,1)
    result=dict(input_files_checked=len(manifest),changed_inputs=changed,npz=datasets,paired=paired,model_or_gpu_used=False)
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'diagnostic_quarantine.json').write_text(json.dumps(dict(purpose='Retrospective diagnosis only. Exclude these tasks, traces and gold patches from all future training and independent final evaluation.',instance_ids=sorted(v14)),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
