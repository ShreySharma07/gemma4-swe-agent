"""Validate downloaded full-training outputs with Python stdlib; no torch/model loading."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import struct


def adapter(path):
    config = json.loads((path / 'adapter_config.json').read_text())
    f = path / 'adapter_model.safetensors'
    with f.open('rb') as handle:
        n = struct.unpack('<Q', handle.read(8))[0]
        assert 0 < n < 100_000_000, 'Invalid safetensors header length'
        header = json.loads(handle.read(n))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        modules = {k.rsplit('.lora_', 1)[0] for k in tensors}
        ranges = sorted(v['data_offsets'] for v in tensors.values())
        cursor = 0
        for begin, end in ranges:
            assert begin == cursor and end >= begin, 'Noncontiguous or overlapping tensor offsets'
            cursor = end
        for v in tensors.values():
            assert v['dtype'] == 'BF16' and len(v['shape']) == 2 and min(v['shape']) == config['r']
            assert v['data_offsets'][1] - v['data_offsets'][0] == 2 * math.prod(v['shape'])
        assert 8 + n + cursor == f.stat().st_size, 'Incomplete adapter file'
        assert len(modules) == 410 and len(tensors) == 820
        assert config['r'] == 16 and config['peft_type'] == 'LORA'
        assert all(m + '.lora_A.weight' in tensors and m + '.lora_B.weight' in tensors for m in modules)
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'status': 'PASS', 'modules': len(modules), 'tensors': len(tensors),
            'bytes': f.stat().st_size, 'sha256': digest.hexdigest()}


def validate(run):
    report = json.loads((run / 'train_report.json').read_text())
    rows = [json.loads(line) for line in (run / 'train_out/metrics.jsonl').read_text().splitlines() if line.strip()]
    plans = [r for r in rows if r.get('event') == 'plan']
    steps = [r for r in rows if r.get('event') == 'step']
    vals = [r for r in rows if r.get('event') == 'val']
    exports = [r for r in rows if r.get('event') == 'export']
    losses = [v['loss'] for v in vals]
    checks = {'return_code_zero': report.get('rc') == 0,
              'all_planned_steps_complete': bool(plans) and len(steps) == plans[0]['optimizer_steps']
              and [s['step'] for s in steps] == list(range(1, plans[0]['optimizer_steps'] + 1)),
              'no_budget_stop': not any(r.get('event') == 'budget_stop' for r in rows),
              'validation_start_and_end': bool(vals) and vals[0]['tag'] == 'start' and vals[-1]['tag'] == 'end',
              'validation_finite': bool(losses) and all(math.isfinite(v) for v in losses),
              'validation_decreases': len(losses) > 1 and losses[-1] < losses[0],
              'validation_no_turn_back_up': len(losses) > 1 and all(b <= a for a, b in zip(losses, losses[1:])),
              'final_410_export': any(e.get('tag') == 'final' and e.get('n_modules') == 410 and e.get('shapes_ok') is True for e in exports),
              'all_export_reports_valid': bool(exports) and all(e.get('n_modules') == 410 and e.get('n_tensors') == 820 and e.get('shapes_ok') is True for e in exports),
              'global_memory_below_21_gb': bool(steps) and all(0 < s['peak_gb'] < 21 for s in steps),
              'throughput_auditable': bool(steps) and all(s.get('tokens', 0) > 0 and s.get('elapsed_s', 0) > 0
               and abs(s['tok_per_s'] - s['tokens'] / s['elapsed_s']) < 0.051 for s in steps)}
    integrity = {}
    for e in exports:
        try:
            integrity[e['tag']] = adapter(run / 'train_out' / ('adapter_' + e['tag']))
        except (AssertionError, OSError, ValueError, KeyError, struct.error) as error:
            integrity[e['tag']] = {'status': 'FAIL', 'error': str(error)}
    try:
        integrity['serving_copy'] = adapter(run / 'adapters' / report['settings']['name'])
    except (AssertionError, OSError, ValueError, KeyError, struct.error) as error:
        integrity['serving_copy'] = {'status': 'FAIL', 'error': str(error)}
    checks['export_files_integrity'] = bool(integrity) and all(v['status'] == 'PASS' for v in integrity.values())
    checks['serving_copy_matches_final'] = integrity.get('final', {}).get('sha256') is not None and integrity['final']['sha256'] == integrity['serving_copy'].get('sha256')
    return {'stage': 'C', 'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'execution_rc': report.get('rc'), 'settings': report.get('settings'), 'plan': plans,
            'optimizer_steps': len(steps), 'validation_progression': vals, 'exports': exports,
            'integrity': integrity, 'resources': {'peak_gb_max': max((s['peak_gb'] for s in steps), default=None),
              'throughput_median': statistics.median(s['tok_per_s'] for s in steps) if steps else None,
              'throughput_range': [min(s['tok_per_s'] for s in steps), max(s['tok_per_s'] for s in steps)] if steps else [],
              'exact_training_tokens': sum(s.get('tokens', 0) for s in steps)},
            'method': 'Stdlib-only metrics and safetensors header/offset/key/shape/config/SHA-256 checks; no tensors or models loaded.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=pathlib.Path)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    result = validate(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'resources': result['resources']}, indent=2))
