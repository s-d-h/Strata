"""tools/dsv41/lora_pack.py - a rank-1 LoRA GGUF (llama.cpp's adapter: general.type adapter, adapter.type lora,
blk.N.attn_output_b.weight / blk.N.ffn_down_shexp.weight .lora_a [n_in, 1] and .lora_b [1, n_out], F32) -> the file
strata-dsv41 --lora reads: the deltas are added to those matrices' outputs as the layers run (W x + scale b (a . x)),
so the pack's weights stay the base's and no BF16 copy of the edited tensors is made.

    python tools/dsv41/lora_pack.py --lora LORA.gguf --out FILE

The file: "DS4L", u32 version 1, u32 count, then per delta u32 layer, u32 kind (0 attn.wo_b, 1 ffn.shared_experts.w2),
u32 n_in, u32 n_out, f32 scale (alpha / rank; alpha 0: 1, llama.cpp's rule), f32 a[n_in], f32 b[n_out]."""
from __future__ import annotations

import argparse
import array
import pathlib
import re
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pack_gguf import Gguf  # noqa: E402

KINDS = {"attn_output_b.weight": 0, "ffn_down_shexp.weight": 1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lora", required=True, help="the LoRA adapter GGUF")
    ap.add_argument("--out", required=True, help="the file strata-dsv41 --lora reads")
    args = ap.parse_args()
    path = pathlib.Path(args.lora)
    g = Gguf(path)
    if g.kv.get("general.type") != "adapter" or g.kv.get("adapter.type") != "lora":
        raise SystemExit("%s: not a LoRA adapter GGUF" % path)
    alpha = float(g.kv.get("adapter.lora.alpha", 0.0))
    infos = {name: (typ, dims, off) for name, typ, dims, off in g.tensors}

    def read(f, name):
        typ, dims, off = infos[name]
        if typ != 0:
            raise SystemExit("%s: type %d, not F32" % (name, typ))
        n = 1
        for d in dims:
            n *= d
        f.seek(g.data_start + off)
        v = array.array("f")
        v.fromfile(f, n)
        return dims, v

    deltas = []
    with path.open("rb") as f:
        for name in sorted(infos):
            if not name.endswith(".lora_a"):
                continue
            base = name[: -len(".lora_a")]
            m = re.fullmatch(r"blk\.(\d+)\.(.+)", base)
            if m is None or m[2] not in KINDS:
                raise SystemExit("%s: not a matrix strata-dsv41 adds a delta to (only %s)" % (base, ", ".join(KINDS)))
            (n_in, ra), a = read(f, name)
            (rb, n_out), b = read(f, base + ".lora_b")
            if ra != 1 or rb != 1:
                raise SystemExit("%s: rank %d / %d: only rank 1" % (base, ra, rb))
            deltas.append((int(m[1]), KINDS[m[2]], n_in, n_out, alpha if alpha else 1.0, a, b))
    with open(args.out, "wb") as f:
        f.write(b"DS4L" + struct.pack("<II", 1, len(deltas)))
        for il, kind, n_in, n_out, scale, a, b in deltas:
            f.write(struct.pack("<IIIIf", il, kind, n_in, n_out, scale))
            a.tofile(f)
            b.tofile(f)
    layers = sorted({d[0] for d in deltas})
    print("%d rank-1 deltas (layers %d-%d, scale %g) -> %s" % (len(deltas), layers[0], layers[-1], deltas[0][4], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
