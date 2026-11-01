"""
k_eq_v adapter fix (stdlib only): copy each global layer's k_proj LoRA to v_proj so vLLM serves V the way it was trained.
Gemma-4's 10 global layers have no v_proj: HF computes V = k_proj(x) (then v_norm), so a k_proj LoRA moves K and V.
vLLM 0.19.1 instead loads the K weights into the V slot of a packed qkv_proj and applies per-slice LoRA (q/k/v), so
with no v_proj LoRA the V slice serves the base K weights only: V silently loses the adapter. Duplicating the
k_proj lora_A/lora_B into v_proj gives the V slice the same delta, i.e. V == K again, as in training.
Global layers are found from the adapter itself: a layer with a k_proj LoRA and no v_proj LoRA.
Usage: python -m evalkit.kvdup <adapter_dir_in> <adapter_dir_out>
Opt-in only: LP1 (vLLM) and LP2 (HF) both score held-out episodes 0.3% better WITHOUT the copy, so adapters ship raw.
"""
import json
import pathlib
import re
import shutil
import struct
import sys

LAYER = re.compile(r"^(.*\.layers\.\d+\.self_attn\.)(k_proj|v_proj)(\.lora_[AB]\.weight)$")


def read(path):
    """-> (header dict, data bytes) of a .safetensors file."""
    raw = pathlib.Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + n]), raw[8 + n:]


def write(path, tensors, meta=None):
    """tensors: {name: (dtype, shape, bytes)} -> .safetensors (contiguous offsets, header padded to 8 bytes)."""
    header, chunks, off = {}, [], 0
    if meta:
        header["__metadata__"] = meta
    for name, (dtype, shape, buf) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(buf)]}
        chunks.append(buf)
        off += len(buf)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for c in chunks:
            f.write(c)


def kv_dup(src_dir, dst_dir):
    """-> report dict. Copies adapter_config.json unchanged (target_modules already lists v_proj)."""
    src, dst = pathlib.Path(src_dir), pathlib.Path(dst_dir)
    header, data = read(src / "adapter_model.safetensors")
    meta = header.pop("__metadata__", None)
    tensors = {k: (v["dtype"], v["shape"], data[v["data_offsets"][0]:v["data_offsets"][1]]) for k, v in header.items()}
    has = {}
    for k in tensors:
        m = LAYER.match(k)
        if m:
            has.setdefault(m.group(1), set()).add(m.group(2))
    glob = sorted((p for p, s in has.items() if s == {"k_proj"}), key=lambda p: int(re.search(r"layers\.(\d+)", p).group(1)))
    for p in glob:
        for ab in (".lora_A.weight", ".lora_B.weight"):
            tensors[p + "v_proj" + ab] = tensors[p + "k_proj" + ab]
    dst.mkdir(parents=True, exist_ok=True)
    write(dst / "adapter_model.safetensors", tensors, meta)
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    mods = {k.rsplit(".lora_", 1)[0] for k in tensors}
    return {"global_layers": [int(re.search(r"layers\.(\d+)", p).group(1)) for p in glob], "added_tensors": 2 * len(glob),
            "n_modules": len(mods), "n_tensors": len(tensors)}


if __name__ == "__main__":
    print(json.dumps(kv_dup(sys.argv[1], sys.argv[2])))
