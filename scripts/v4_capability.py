#!/usr/bin/env python3
"""Capability retention + throughput (PREREG_v4.md sections 4-5).

Logit-scored multiple choice (as Deviation 3): 300 English MMLU test items + 300 Global-MMLU Hindi test items,
sampled once with seed 20261115; raw prompt "Question/प्रश्न ... Answer/उत्तर:"; prediction = argmax over A-D
letter logits. Arms: base, base+LoRA v3, base+LoRA v4 (if selected). Wilson 95% CI.
Throughput: HF generate, greedy, 128 new tokens, on 64 benign fresh-test prompts, batch 1 and 16, base vs adapter.
Writes results/v4_fresh/capability.json.
"""
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_v3_common import ROOT, chat_prompt, free, load_generator, log, read_jsonl  # noqa: E402

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
OUT = ROOT / "results/v4_fresh"
N = 300


def wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def load_items():
    rng = random.Random(20261115)
    en = pd.concat([pd.read_parquet(f) for f in glob.glob(str(HUB / "datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))])
    hi = pd.concat([pd.read_parquet(f) for f in glob.glob(str(HUB / "datasets--CohereLabs--Global-MMLU/snapshots/*/hi/test-*.parquet"))])
    en_i = rng.sample(range(len(en)), N)
    hi_i = rng.sample(range(len(hi)), N)
    E = [dict(p=f"Question: {r.question}\nA. {r.choices[0]}\nB. {r.choices[1]}\nC. {r.choices[2]}\nD. {r.choices[3]}\nAnswer:", a="ABCD"[int(r.answer)])
         for r in en.iloc[en_i].itertuples()]
    H = [dict(p=f"प्रश्न: {r.question}\nA. {r.option_a}\nB. {r.option_b}\nC. {r.option_c}\nD. {r.option_d}\nउत्तर:", a=str(r.answer).strip().upper())
         for r in hi.iloc[hi_i].itertuples()]
    return E, H


def mc(tok, model, items):
    ids = {c: (tok.encode(" " + c, add_special_tokens=False)[-1], tok.encode(c, add_special_tokens=False)[-1]) for c in "ABCD"}
    k = 0
    for it in items:
        enc = tok(it["p"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1].float()
        pred = max("ABCD", key=lambda c: max(lg[ids[c][0]].item(), lg[ids[c][1]].item()))
        k += pred == it["a"]
    return dict(acc=round(k / len(items), 4), ci=wilson(k, len(items)), n=len(items))


def throughput(tok, model, prompts):
    out = {}
    for bs in [1, 16]:
        texts = [chat_prompt(tok, p) for p in prompts[: (16 if bs == 1 else 64)]]
        tok.padding_side = "left"
        n_tok, t0 = 0, None
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=128, min_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
            n_tok += (g.shape[1] - enc["input_ids"].shape[1]) * g.shape[0]
        torch.cuda.synchronize()
        dt = time.time() - t0
        out[f"batch_{bs}"] = dict(new_tokens=n_tok, seconds=round(dt, 2), tokens_per_s=round(n_tok / dt, 1), replies_per_s=round(len(texts) / dt, 3))
    return out


def main():
    E, H = load_items()
    sel = json.load(open(ROOT / "results/lora_v4/validation_selection.json")) if (ROOT / "results/lora_v4/validation_selection.json").exists() else {}
    arms = [("raw_baseline", None), ("lora_v3", ROOT / "up/checkpoint-150")]
    if sel.get("status") == "SELECTED":
        arms.append(("lora_v4", ROOT / "results/lora_v4/adapter" / f"checkpoint-{sel['selected'].split('_')[1]}"))
    ps = ROOT / "results/probe_steering"
    steer = None
    if (ps / "selection.json").exists() and json.load(open(ps / "selection.json")).get("status") == "SELECTED":
        steer = torch.load(ps / "selected_config.pt")
        arms.append(("base_gated_steering", None))
    benign = [r for r in read_jsonl(ROOT / "data/v4_fresh/test.jsonl") if r["intent"] == "benign"]
    random.Random(1).shuffle(benign)
    res = dict(method=__doc__.strip().splitlines()[0], gpu=torch.cuda.get_device_name(0), arms={}, throughput={})
    for name, ad in arms:
        tok, model = load_generator(ad)
        model.eval()
        gs = None
        if name.endswith("gated_steering"):
            from steering_gate import make_steered
            gs, backend = make_steered(model, steer, gated=True)
            res.setdefault("steering_backend", backend)
            model = gs
        res["arms"][name] = dict(mmlu_en=mc(tok, model, E), gmmlu_hi=mc(tok, model, H))
        if name in ("raw_baseline", "lora_v3", "lora_v4"):
            res["throughput"][name] = throughput(tok, model, benign)
        if gs:
            res["arms"][name]["gate_fired_on_mc"] = [gs.n_fired, gs.n_seen]
            gs.detach()
            model = gs.model
        log(f"{name}: {res['arms'][name]} {res['throughput'].get(name)}")
        free(model)
    json.dump(res, open(OUT / "capability.json", "w"), indent=1)
    log("CAPABILITY DONE")


if __name__ == "__main__":
    main()
