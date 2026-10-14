"""Audit legacy variable-window training rates with stdlib only; never import trainkit/torch.

Raw logs are unchanged. A one-decimal rate bounds each rank's original elapsed
time; total tokens / maximum rank elapsed time then bounds global throughput.
"""
import argparse
import ast
import hashlib
import json
import pathlib
import random
import statistics
import struct
import zipfile


def array(z, name):
    b = z.read(name + '.npy')
    assert b[:6] == b'\x93NUMPY'
    offset = 10 if b[6] == 1 else 12
    n = struct.unpack('<H' if offset == 10 else '<I', b[8:offset])[0]
    header = ast.literal_eval(b[offset:offset + n].decode())
    assert not header['fortran_order'] and len(header['shape']) == 1
    formats = {'<i8': 'q', '<i4': 'i', '|u1': 'B', '|b1': '?', '<u8': 'Q'}
    values = [v[0] for v in struct.iter_unpack('<' + formats[header['descr']], b[offset + n:])]
    assert len(values) == header['shape'][0]
    return values


def elapsed_bounds(tokens, world, rounded_rate):
    assert tokens > 0 and rounded_rate > 0.05
    return (world * tokens / (rounded_rate + 0.05),
            world * tokens / (rounded_rate - 0.05))


def global_bounds(tokens, rates):
    times = [elapsed_bounds(t, len(tokens), r) for t, r in zip(tokens, rates)]
    return sum(tokens) / max(t[1] for t in times), sum(tokens) / max(t[0] for t in times)


def audit(run):
    config = json.loads((run / 'train_report.json').read_text())['settings']
    original_worker = run / 'trainkit/worker.py'
    original_data = run / 'trainkit/data.py'
    assert 'tok_per_s=round(self.world * tok / dt, 1)' in original_worker.read_text()
    assert 'group = sorted(reps[i:i + world], key=self.length)' in original_data.read_text()
    data = run / 'train_data.npz'
    with zipfile.ZipFile(data) as z:
        starts, val = array(z, 'starts'), array(z, 'val')
    windows = list(range(len(starts) - 1))
    random.Random(config['seed']).shuffle(windows)
    chosen = [w for w in windows if not val[w]][:config['max_windows']]
    length = lambda w: starts[w + 1] - starts[w]
    files = sorted((run / 'train_out/rank_logs').glob('**/stdout.log'), key=lambda p: int(p.parent.name))
    world = len(files)
    assert world == 4 and config['epochs'] == 1
    groups = [sorted(chosen[i:i + world], key=length) for i in range(0, len(chosen) - world + 1, world)]
    logs = []
    for rank, f in enumerate(files):
        assert f.parent.name == str(rank)
        rows = [json.loads(line.split(' ', 1)[1]) for line in f.read_text().splitlines() if line.startswith(f'rank{rank} ')]
        plan = next(r for r in rows if r['event'] == 'plan')
        assert plan['train_tokens'] == sum(map(length, chosen))
        assert plan['micro_steps'] == len(groups)
        logs.append({r['step']: r for r in rows if r['event'] == 'step'})
    expected_steps = (len(groups) + config['grad_accum'] - 1) // config['grad_accum']
    assert all(sorted(rows) == list(range(1, expected_steps + 1)) for rows in logs)
    metrics = [json.loads(l) for l in (run / 'train_out/metrics.jsonl').read_text().splitlines()]
    assert [r for r in metrics if r['event'] == 'step'] == list(logs[0].values())
    results = []
    accum = config['grad_accum']
    for i in range(0, len(groups), accum):
        step = i // accum + 1
        group = groups[i:i + accum]
        tokens = [sum(length(g[rank]) for g in group) for rank in range(world)]
        rates = [r[step]['tok_per_s'] for r in logs]
        bounds = [elapsed_bounds(t, world, r) for t, r in zip(tokens, rates)]
        low, high = global_bounds(tokens, rates)
        results.append({'step': step, 'window_ids_by_microstep': group,
                        'rank_tokens': tokens, 'total_tokens': sum(tokens),
                        'legacy_rank_rates': rates, 'rank_elapsed_s_bounds': bounds,
                        'corrected_tok_per_s_bounds': [low, high],
                        'within_target_for_entire_interval': 240 <= low <= high <= 420})
    assert sum(r['total_tokens'] for r in results) == sum(map(length, chosen))
    return {'method': 'Exact token counts reconstructed from preserved NPZ and seed-0 schedule; elapsed bounds inferred from all four original rates rounded to 0.1. Corrected rate uses maximum rank elapsed time. Absolute exact elapsed times were not recorded.',
            'sources': {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [data, run / 'train_report.json', run / 'train_out/metrics.jsonl', original_worker, original_data, *files]},
            'total_training_tokens': sum(map(length, chosen)), 'steps': results,
            'corrected_median_bounds': [statistics.median(r['corrected_tok_per_s_bounds'][j] for r in results) for j in (0, 1)],
            'status': 'PASS' if all(r['within_target_for_entire_interval'] for r in results) else 'FAIL'}


def self_test():
    # Known uneven workload: legacy rank0 rate 100, actual global rate 250.
    low, high = global_bounds([100, 200, 300, 400], [100, 200, 300, 400])
    assert low < 250 < high and high - low < 0.3
    # Slow rank must control the denominator, not rank0's shorter duration.
    low, high = global_bounds([100, 100], [200, 100])
    assert low < 100 < high and high < 101
    low, high = global_bounds([100, 100], [200, 200])
    assert low < 200 < high
    print('PASS: uneven workload, slow-rank timing, equal workload')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=pathlib.Path, nargs='?')
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        result = audit(args.run)
        serialized = json.dumps(result, indent=2) + '\n'
        if args.output:
            args.output.write_text(serialized)
        print(json.dumps({'status': result['status'], 'total_tokens': result['total_training_tokens'],
                          'step_bounds': [s['corrected_tok_per_s_bounds'] for s in result['steps']],
                          'median_bounds': result['corrected_median_bounds']}, indent=2))
