"""Read-only, stdlib-only ATIF evidence audit. Never executes trace commands.
Writes derived research ONLY beside this script. Counts proposals separately from
response association; mutation recognition is conservative intent, not proof.
"""
import ast
import collections as C
import hashlib
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'data/runs/atul_eval_v14_new60/x/eval_v14_thinking/rep0'

def parse_content(value):
    if isinstance(value, dict):
        return value
    try:
        x = json.loads(value or '{}')
        return x if isinstance(x, dict) else {}
    except (ValueError, TypeError):
        return {}

def kind(name, args):
    cmd = str(args.get('command', ''))
    if name in ('write_file', 'edit_file'):
        return 'mutation_intent'
    if name != 'run_command':
        return 'graph' if name in ('get_code_neighbors','get_code_subgraph','search_similar_code') else name
    # Requires actual write syntax, not a read_text or comparison operator.
    if re.search(r'\.(write_text|write_bytes)\s*\(|\bopen\([^\n]*,[ \t]*[\'\"](?:w|a|x)[b+]?|\bsed\s+-i\b|\b(?:apply_patch|patch)\b', cmd):
        return 'mutation_intent'
    clean = re.sub(r'^#[^\n]*(?:\n|$)', '', cmd.strip(), flags=re.M).strip()
    if re.search(r'(?:^|&&|;|\n)\s*(?:python[\d.]*\s+-m\s+)?pytest\b|(?:^|&&|;|\n)\s*python[\d.]*\s+-m\s+unittest\b',clean):
        return 'test'
    if re.search(r'(?:^|&&|;|\n)\s*python[\d.]*\s+(?:repro\w*|verify\w*|test_\w*)\.py\b',clean):
        return 'probe_script'
    if re.search(r'\b(?:grep|rg|find)\b',cmd):
        return 'search'
    if re.search(r'\b(?:sed\s+-n|cat|head|tail|ls)\b|\.read_text\(',cmd):
        return 'read'
    return 'other'

def mutation_paths(name, args):
    """Literal mutation targets only; unknown/dynamic paths stay unknown."""
    if name in ('write_file','edit_file'):
        return [str(args['filepath'])] if args.get('filepath') else []
    cmd=str(args.get('command','')); paths=[]
    # AST parse heredoc bodies, never eval/exec. Only a common, explicit Path form.
    blocks=re.findall(r"<<[ \t]*['\"]?(\w+)['\"]?[^\n]*\n(.*?)(?:\n\1(?:\n|$))",cmd,re.S)
    for _,body in blocks:
        try:tree=ast.parse(body)
        except SyntaxError:continue
        aliases={}
        def literal_path(node):
            if isinstance(node,ast.Name):return aliases.get(node.id)
            if isinstance(node,ast.Call) and ((isinstance(node.func,ast.Attribute) and node.func.attr=='Path') or (isinstance(node.func,ast.Name) and node.func.id=='Path')) and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):return node.args[0].value
            return None
        for node in tree.body:
            if isinstance(node,ast.Assign):
                value=literal_path(node.value)
                for target in node.targets:
                    if isinstance(target,ast.Name) and value:aliases[target.id]=value
            for child in ast.walk(node):
                if isinstance(child,ast.Call) and isinstance(child.func,ast.Attribute) and child.func.attr in ('write_text','write_bytes'):
                    value=literal_path(child.func.value)
                    if value:paths.append(value)
    return sorted(set(paths))

def path_class(path):
    path=path.removeprefix('/workspace/').removeprefix('./')
    if path.startswith(('fastapi/','rich/','src/requests/','docs_src/','docs/')):return 'repository'
    if path.startswith(('tests/','/tmp/')) or re.match(r'(?:repro|reproduce|verify|test_|output)',path):return 'scratch_or_test'
    return 'unknown'

def sequence(trace):
    """ID association first; FIFO same-tool fallback explicitly labelled."""
    events, pending, unmatched = [], [], 0
    for step in trace.get('steps', []):
        for call in step.get('tool_calls') or []:
            args = call.get('arguments') or {}
            if not isinstance(args,dict):
                args = parse_content(args)
            e = dict(step=step.get('step_id'), name=call.get('function_name'),
                     cid=call.get('tool_call_id'), args=args, response=None,
                     association='missing', kind=kind(call.get('function_name'),args), mutation_paths=mutation_paths(call.get('function_name'),args))
            events.append(e);pending.append(e)
        obs=step.get('observation') or {}
        for res in obs.get('results') or ([obs] if obs else []):
            cid=res.get('source_call_id') or res.get('tool_call_id')
            name=(res.get('extra') or obs.get('extra') or {}).get('tool_name')
            candidates=[e for e in pending if e['cid']==cid] if cid else [e for e in pending if not name or e['name']==name]
            if not candidates:
                unmatched+=1;continue
            e=candidates[0];pending.remove(e)
            e['response']=parse_content(res.get('content'))
            e['raw_response']=str(res.get('content',''))
            e['association']='id' if cid else 'ordered_same_tool'
    return events,unmatched

def test_state(response):
    if response is None: return 'unknown'
    details=response.get('details') or {}
    err=str(response.get('error','')); text=str(response.get('stdout',details.get('stdout','')))+' '+str(response.get('stderr',details.get('stderr','')))+' '+err
    if 'BudgetExceeded' in str(response): return 'denied'
    if re.search(r'ModuleNotFoundError|ImportError|ERROR collecting|No module named',text): return 'environment_or_collection'
    if re.search(r'\bFAILED\b|\b[1-9]\d* failed\b',text): return 'failed'
    if re.search(r'\b[1-9]\d* passed\b',text): return 'passed'
    # Successful shell exit can hide pipe failures; it is not a test pass oracle.
    return 'unknown'

def stats(trace):
    es,unmatched=sequence(trace)
    trans=C.Counter();seen_edit=False
    for i,e in enumerate(es):
        if e['kind']=='test' and i+1<len(es):
            trans[('post_mutation' if seen_edit else 'pre_mutation')+':'+test_state(e['response'])+'->'+es[i+1]['kind']]+=1
        if e['kind']=='mutation_intent':seen_edit=True
    return es,unmatched,trans

def main():
    analyses={r['instance_id']:r for r in map(json.loads,(RUN/'analysis/per_task.jsonl').read_text().splitlines())}
    rows=[];counts=C.Counter();trans=C.Counter()
    for r in map(json.loads,(RUN/'merged/task_results.jsonl').read_text().splitlines()):
        iid=r['instance_id'];p=RUN/'merged/traces'/('trace_'+iid+'.json');trace=json.loads(p.read_text());a=analyses[iid]
        es,unmatched,t=stats(trace);trans.update(t);counts.update(e['kind'] for e in es)
        metrics=[dict(step=s['step_id'],**s['metrics']) for s in trace['steps'] if s.get('metrics')]
        bodies=[e.get('raw_response','') for e in es];seen=set();duplicate_chars=0
        for b in bodies:
            if b and b in seen:duplicate_chars+=len(b)
            seen.add(b)
        muts=[i for i,e in enumerate(es) if e['kind']=='mutation_intent']
        source_muts=[i for i,e in enumerate(es) if any(path_class(p)=='repository' for p in e['mutation_paths'])]
        patchpath=RUN/'merged/patches'/(iid+'.patch');patch=patchpath.read_text() if patchpath.exists() else ''
        final_log=RUN/'merged/test_outputs'/(iid+'.log');log=final_log.read_text() if final_log.exists() else ''
        errors=C.Counter(str(e['response'].get('error_type','unknown')) for e in es if e['response'] and e['response'].get('status')=='error')
        # Error schemas differ: retain canonical observed error-type counts separately.
        errors=C.Counter(x for e in es for x in ['BudgetExceeded','CommandError'] if x in e.get('raw_response',''))
        row=dict(**r, trace=str(p.relative_to(ROOT)), trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            old_failure_stage=a['failure_stage'],gold_files=a['gold_files'],patch_files=a['patch_files'],
            gold_file_seen=a['step_first_gold_file'] is not None,gold_lines_seen=a['step_first_gold_lines'] is not None,
            patch_touches_gold=a['patch_touches_gold'],proposed_calls=len(es),response_associations=dict(C.Counter(e['association'] for e in es)),unmatched_observations=unmatched,
            mutation_intents=len(muts),repository_mutation_intents=len(source_muts),mutation_paths=sorted({p for e in es for p in e['mutation_paths']}),
            tests_or_probes_after_last_repository_mutation=sum(e['kind'] in ('test','probe_script') for e in es[source_muts[-1]+1:]) if source_muts else None,mutation_response_ok=sum(e['kind']=='mutation_intent' and (e['response'] or {}).get('status')=='ok' for e in es),
            tests_after_last_mutation=sum(e['kind']=='test' for e in es[muts[-1]+1:]) if muts else None,
            tests=[dict(step=e['step'],state=test_state(e['response'])) for e in es if e['kind']=='test'],
            errors=dict(errors),actions=dict(C.Counter(e['kind'] for e in es)),transitions=dict(t),
            explicit_submit=any(e['name']=='submit_patch' for e in es),
            repeated_calls=len(es)-len({json.dumps([e['name'],e['args']],sort_keys=True) for e in es}),
            compactions=a['compactions'],context_resets=a['context_resets'],metrics=metrics,
            max_prompt=max([m.get('prompt_tokens',0) for m in metrics] or [0]),
            completion_tokens=sum(m.get('completion_tokens',0) for m in metrics),
            thought_tokens=None,
            candidate_stream_chars_saved_at_1200=sum(max(0,len(str((e['response'] or {}).get(key,((e['response'] or {}).get('details') or {}).get(key,''))))-1200) for e in es for key in ('stdout','stderr')),
            tool_response_chars=sum(map(len,bodies)),max_tool_response_chars=max(map(len,bodies),default=0),
            duplicate_response_chars=duplicate_chars,truncation_markers=sum(bool(re.search('truncat',b,re.I)) for b in bodies),
            wrapt_warning_present='wrapt' in log,
            verifier_diagnostics=list(dict.fromkeys(re.findall(r'^E\s+(?:AssertionError|TypeError|AttributeError|ImportError|ModuleNotFoundError|SyntaxError)[^\n]*',log,re.M)))[:12],
            patch_added=sum(l.startswith('+') and not l.startswith('+++') for l in patch.splitlines()),
            patch_removed=sum(l.startswith('-') and not l.startswith('---') for l in patch.splitlines()),
            event_summary=[dict(step=e['step'],kind=e['kind'],association=e['association'],status=(e['response'] or {}).get('status'),test_state=test_state(e['response']) if e['kind']=='test' else None) for e in es])
        rows.append(row)
    assert len(rows)==60 and len({r['instance_id'] for r in rows})==60
    (OUT/'trace_audit.json').write_text(json.dumps(dict(tasks=rows,action_counts=dict(counts),transitions=dict(trans)),indent=2)+'\n')
    datasets={}
    for name,archive in [('Step C','step_c_preprocessing_20261016.tar.gz'),('S3b','s3b_episodes_20261018.tar.gz')]:
        ds_counts=C.Counter();ds_trans=C.Counter();first=C.Counter();digests=C.Counter();repos=C.Counter();n=0;association=C.Counter()
        with tarfile.open(ROOT/'data/runs'/archive) as tar:
            for m in tar:
                if not m.isfile() or not Path(m.name).name.startswith('trace_') or not m.name.endswith('.json'):continue
                data=tar.extractfile(m).read();tr=json.loads(data);es,unmatched,ts=stats(tr);n+=1
                ds_counts.update(e['kind'] for e in es);ds_trans.update(ts);first.update([es[0]['kind']] if es else []);association.update(e['association'] for e in es)
                canonical=json.dumps([[e['name'],e['args'],e['response']] for e in es],sort_keys=True)
                digests[hashlib.sha256(canonical.encode()).hexdigest()]+=1
                parts=Path(m.name).stem.removeprefix('trace_').split('__');repos['/'.join(parts[:2]).rsplit('-',1)[0]]+=1
        datasets[name]=dict(episodes=n,action_counts=dict(ds_counts),first_action=dict(first),transitions=dict(ds_trans),association=dict(association),exact_duplicate_episodes=sum(v-1 for v in digests.values()),repository_keys_from_filenames=len(repos),top_repository_keys=repos.most_common(8))
    (OUT/'dataset_measurements.json').write_text(json.dumps(datasets,indent=2)+'\n')
    print(json.dumps(dict(tasks=len(rows),resolved=sum(r['resolved'] for r in rows),context_errors=sum('context' in (r['error'] or '').lower() for r in rows),actions=dict(counts),datasets={k:{'episodes':v['episodes'],'first_action':v['first_action']} for k,v in datasets.items()}),indent=2))

if __name__=='__main__':main()
