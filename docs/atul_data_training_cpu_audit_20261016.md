# Atul Baliyan — lightweight preprocessing evidence audit

October 16, 2026. This extends the completed Step D diagnosis with a read-only NPZ supervision-weight audit. It does not repeat the 114-task forensic analysis or decode training actions. Only Python standard-library archive/binary parsing was executed; no NumPy, PyTorch, model, GPU, saved trajectory command or repository replay was run.

## Results

Input: `data/runs/train_public_200.npz`, SHA-256 `b3f51bc449c31de2a3b163f98c5fdec2725b4427bde15625af7b4c75447d2488`.

| Measurement | Result |
|---|---:|
| Windows / input tokens | 1,017 / 7,372,758 |
| Window length min / median / max | 2,799 / 7,661 / 8,191 |
| Shifted supervised labels min / median / max | 23 / 895 / 4,726 |
| Total shifted supervised labels | 1,170,308 |
| Masked-in first-position tokens | 0 |
| Windows with 1–99 / 100–499 / 500–999 labels | 49 / 365 / 109 |
| Windows with 1,000–1,999 / ≥2,000 labels | 273 / 221 |
| Scheduled training windows, seed 0, four ranks | 972 |
| Scheduled supervised labels | 1,114,033 |
| Scheduled label count min / median / max | 23 / 879 / 4,726 |
| Largest/smallest within-window token coefficient ratio | 4,726 / 23 = 205.4783 |

Agent 2 computed these values, and the primary researcher independently reproduced them. The last ratio describes nominal CE coefficients under equal-window mean loss, not measured gradient influence. No semantic action categories can be assigned without the historical tokenizer and original rendered examples. In particular, a short window must not be assumed to be a submission window.

Under the inspected `[:8192]` tokenizer cap, any clipped sequence would be exactly 8,192 tokens. None is. Thus that clipping cannot explain C conditional on matching preprocessing-source provenance. Missing earlier state, character output trimming and boundary masking remain separate questions.

## Reproduction

Run from repository root using a standard-library Python interpreter. This reads one NPZ and prints aggregate statistics only. Native binary array interpretation below assumes a little-endian machine, as used for this audit.

```python
import array, ast, hashlib, io, json, random, statistics, struct, sys, zipfile
from pathlib import Path

assert sys.byteorder == "little"
p = Path("data/runs/train_public_200.npz")

def read_npy(data):
    f = io.BytesIO(data)
    assert f.read(6) == b"\x93NUMPY"
    major, minor = f.read(2)
    size = 2 if major == 1 else 4
    n = struct.unpack("<H" if size == 2 else "<I", f.read(size))[0]
    header = ast.literal_eval(f.read(n).decode())
    raw = f.read()
    kind = header["descr"]
    if kind.endswith("i8"):
        return list(array.array("q", raw))
    if kind.endswith("u1"):
        return list(raw)
    raise ValueError(kind)

with zipfile.ZipFile(p) as z:
    starts = read_npy(z.read("starts.npy"))
    mask = read_npy(z.read("train.npy"))
    val = read_npy(z.read("val.npy"))

lens = [b-a for a, b in zip(starts, starts[1:])]
labels = [sum(mask[a+1:b]) for a, b in zip(starts, starts[1:])]
order = list(range(len(lens)))
random.Random(0).shuffle(order)
training = [i for i in order if not val[i]]
scheduled = training[:len(training)//4*4]
def mmm(values):
    return [min(values), statistics.median(values), max(values)]
print(json.dumps({
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "windows": len(lens), "input_tokens": sum(lens),
    "length_min_median_max": mmm(lens),
    "supervised_min_median_max": mmm(labels),
    "supervised_total": sum(labels),
    "first_position_masked_in": sum(mask[a] for a in starts[:-1]),
    "supervised_histogram": [
        sum(lo <= n < hi for n in labels)
        for lo, hi in [(1,100),(100,500),(500,1000),(1000,2000),(2000,float("inf"))]
    ],
    "scheduled_windows": len(scheduled),
    "scheduled_supervised": sum(labels[i] for i in scheduled),
    "scheduled_min_median_max": mmm([labels[i] for i in scheduled]),
    "nominal_coefficient_ratio": max(labels)/min(labels),
}, indent=2))
```

Length-sorting within each group assigns ranks but does not change scheduled membership, so the audit reproduces the aggregate schedule without importing the training module.

## Source identity

These are inspected current-source hashes, not proof of historical preprocessing provenance.

| File | SHA-256 |
|---|---|
| `posttrain/convert_public.py` | `0c4853833876a2ea48ed7f24427da99d3b4a219fbd3369d745788888ef4d7a06` |
| `posttrain/render_atif.py` | `ac3b2edaf2c668c4902e1a54820c13e89b026ab3a91ed66289b925409f90ef6d` |
| `posttrain/tokenize_windows.py` | `6070dc4d36c0c783215914eee25e7e57dcf4beba971a9efa32154ebfcfcf06df` |
| `kaggle/train/trainkit/data.py` | `4150786d8e1c785ed9a3c43b92fc46bb8f6f84dd5d46088b1482cfd8b560651b` |
| `kaggle/train/trainkit/model.py` | `c62bc225816094e1938cc4aa34b0c6c587f022eaf73e4d64bdce04e59cecbfee` |
| `kaggle/train/trainkit/worker.py` | `ed2339c27c02c3d5aaf195ed34042c03a26cb1586345e4a51ad01b3000d46763` |

## Research evidence records

Completed delegated research: [dataset design](research/atul_data_training_20261016/dataset_notes.md), [pipeline/supervision](research/atul_data_training_20261016/pipeline_notes.md), [training methods](research/atul_data_training_20261016/methods_notes.md). These are working evidence memos; the [main report](atul_data_training_research_20261016.md) is the reviewed recommendation.

Literature and synthesis delegates hit the account usage limit. The primary researcher completed both passes, reading the complete diagnosis first. Literature checks used original arXiv papers, official repositories and the source dataset card. Verified sources/results and limitations are retained in main-report §5. No independent completion by those two delegates is claimed.

The synthesis pass rejected: treating plausible conversion failures as proven C defects; assigning low edit-token share without decoding; hard clipping as a leading C cause despite the 8,191 maximum; online RL within 2.88 remaining quota-hours; new summary formats without serving support; and interpreting 4/4 toy fixtures as generalization evidence. It retained Paths A→B→conditional C with data fidelity, causal state and behavioral gates. The minimum next experiment uses zero GPU hours. No corrective implementation was performed.
