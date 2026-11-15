#!/usr/bin/env python3
"""EXPLORATORY step 3: refusal components (CircuitKIT attribution + mean-ablation causal check). Dev prompts only.

Attribution: CircuitKIT 0.2.4 `multilingual_refusal` (EAP-IG, node level = attention heads + MLPs), chat template ON,
20 minimal harmful/harmless pairs per form. Openers = the dominant tokens of our refusal / compliance sets ("I'm" vs a
newline), the same for every form, so the CircuitKIT metric approximates our set-based refusal logit-diff.
English pairs: bundled with CircuitKIT. Romanized / Hinglish: the same pairs, each translated together by Sarvam-M.
Scores: the `*_scores.pt` side-car (Circuit.top_nodes() returns 1.0 for every node in 0.2.4).

Causal check (HF model, our set-based metric, the 60 dev harmful scenarios per form): mean-ablate the form's top-k
components (k = 5, 10, 20) vs k random components drawn from the same layers (5 seeds). Mean = per-component mean
activation over all non-pad positions of that form's dev prompts. Heads: the head's slice of the o_proj input.
MLPs: the MLP output. Writes results/refusal_localisation/step3.json.
"""
import csv
import json
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_ID, MODEL_REV, ROOT, free, load_generator, log, read_jsonl  # noqa: E402
import refusal_localisation as RL  # noqa: E402

OUT = ROOT / "results/refusal_localisation"
N_PAIRS = 20
OPENERS = ("I'm", "\n")
FORMS = ["English", "Romanized Hindi", "Hinglish"]


def translate_pairs(en_pairs):
    import v4_build_fresh_test as b
    llm = b.LLM(b.TRANSLATOR)
    out = {}
    for form in ["Romanized Hindi", "Hinglish"]:
        convs = [[{"role": "system", "content": b.SYS_TRANSLATE},
                  {"role": "user", "content": f"{b.FORM_INSTR[form]}\nTranslate BOTH lines below. They differ in only one or two words; keep every other "
                                              f"word identical between the two translations. Output exactly two lines: the first translation, then the second.\n\n{h}\n{s}"}]
                 for h, s in en_pairs]
        pairs = []
        for r in llm.chat(convs, max_new_tokens=200):
            ls = [x.strip() for x in r.splitlines() if x.strip()]
            if len(ls) >= 2 and ls[0] != ls[1]:
                pairs.append((ls[0], ls[1]))
        out[form] = pairs
    free(llm)
    return out


def attribution():
    import circuitkit as ck
    from circuitkit.tasks.builtins import multilingual_refusal as M
    en = M.load_pairs({"language": "en"})[:60]
    tr = translate_pairs(en[:N_PAIRS * 2])
    model = ck.load_model(MODEL_ID, revision=MODEL_REV, dtype="bfloat16", device="cuda")
    res = {}
    for form in FORMS:
        dp = dict(language="en" if form == "English" else "hi-Latn", num_examples=N_PAIRS, seed=0, data_partition="all",
                  refuse=OPENERS[0], comply=OPENERS[1])
        if form != "English":
            p = OUT / f"step3_pairs_{form.replace(' ', '_')}.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["harmful", "harmless"])
                w.writerows(tr[form])
            dp["pairs_path"] = str(p)
        stem = OUT / f"step3_ck_{form.replace(' ', '_')}.pt"
        ck.discover(model, "multilingual_refusal", algorithm="eap-ig", level="node", n_examples=N_PAIRS, batch_size=4, sparsity=0.3,
                    scope="both", output_path=str(stem), data_params=dp)
        sc = torch.load(str(stem).replace(".pt", "_scores.pt"), map_location="cpu", weights_only=False)["node_scores"]
        ranked = sorted(((str(k), float(v)) for k, v in sc.items()), key=lambda kv: abs(kv[1]), reverse=True)
        res[form] = dict(top20=[(k, round(v, 6)) for k, v in ranked[:20]], n_nodes=len(ranked),
                         pairs=N_PAIRS if form == "English" else len(tr[form]))
    free(model)
    tops = {f: {k for k, _ in res[f]["top20"]} for f in FORMS}
    overlap = {f"{a}|{b}": dict(shared=len(tops[a] & tops[b]), jaccard=round(len(tops[a] & tops[b]) / len(tops[a] | tops[b]), 3))
               for i, a in enumerate(FORMS) for b in FORMS[i + 1:]}
    return res, overlap


def parse(c):
    m = re.match(r"A(\d+)\.(\d+)$", c)
    if m:
        return ("head", int(m.group(1)), int(m.group(2)))
    m = re.match(r"MLP (\d+)$", c)
    return ("mlp", int(m.group(1)), None) if m else None


def causal(attr):
    dev = read_jsonl(RL.DEV)
    ev = [r for r in dev if r["role"] == "eval_harmful"]
    S = json.load(open(OUT / "summary.json"))
    R = [int(t) for t in S["token_sets"]["refusal_tokens"]]
    C = [int(t) for t in S["token_sets"]["compliance_tokens"]]
    tok, model = load_generator(None)
    run = RL.Runner(tok, model)
    run.R, run.C = R, C
    layers = run.layers
    nh = model.config.num_attention_heads
    hd = model.config.hidden_size // nh

    def means(items):
        """mean o_proj input per (layer) and mean MLP output per layer over non-pad positions."""
        acc_o, acc_m, n = defaultdict(lambda: 0), defaultdict(lambda: 0), 0
        for i in range(0, len(items), 16):
            enc = run.enc(items[i:i + 16])
            mask = enc["attention_mask"].bool()
            hs = []
            for L, lay in enumerate(layers):
                hs.append(lay.self_attn.o_proj.register_forward_pre_hook(lambda m, a, L=L: acc_o.__setitem__(L, acc_o[L] + a[0][mask].float().sum(0))))
                hs.append(lay.mlp.register_forward_hook(lambda m, a, o, L=L: acc_m.__setitem__(L, acc_m[L] + o[mask].float().sum(0))))
            with torch.no_grad():
                model(**enc)
            for h in hs:
                h.remove()
            n += int(mask.sum())
        return {L: v / n for L, v in acc_o.items()}, {L: v / n for L, v in acc_m.items()}

    def ablate(comps, mo, mm):
        by_head, mlps = defaultdict(list), set()
        for c in comps:
            t, L, h = parse(c)
            (by_head[L].append(h) if t == "head" else mlps.add(L))

        def hooks(i, j):
            hs = []
            for L, heads in by_head.items():
                def pre(m, a, L=L, heads=heads):
                    x = a[0].clone()
                    for h in heads:
                        x[..., h * hd:(h + 1) * hd] = mo[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,) + tuple(a[1:])
                hs.append(layers[L].self_attn.o_proj.register_forward_pre_hook(pre))
            for L in mlps:
                hs.append(layers[L].mlp.register_forward_hook(lambda m, a, o, L=L: torch.zeros_like(o) + mm[L].to(o.dtype)))
            return hs
        return hooks

    out = {}
    for form in FORMS:
        items = sorted([r for r in ev if r["language"] == form], key=lambda r: r["scenario_id"])
        mo, mm = means(items)
        base = run.ld(run.logits_last(items))
        top = [k for k, _ in attr[form]["top20"] if parse(k)]
        fr = dict(base=float(base.mean()), k={})
        for k in (5, 10, 20):
            comps = top[:k]
            v = run.ld(run.logits_last(items, ablate(comps, mo, mm)))
            rnd = []
            for s in range(5):
                rng = random.Random(100 * s + k)
                pick = []
                for c in comps:
                    t, L, h = parse(c)
                    pool = [f"A{L}.{x}" for x in range(nh)] + [f"MLP {L}"]
                    pool = [p for p in pool if p not in comps and p not in pick]
                    pick.append(rng.choice(pool))
                rnd.append(float(base.mean() - run.ld(run.logits_last(items, ablate(pick, mo, mm))).mean()))
            by = defaultdict(list)
            for r, a, b in zip(items, base, v):
                by[r["scenario_id"]].append(float(a - b))
            fr["k"][k] = dict(top_drop=RL.boot_mean(by), random_drop_mean=float(np.mean(rnd)), random_drop_sd=float(np.std(rnd)), random_drops=rnd)
        out[form] = fr
    free(model)
    return out


def main():
    t0 = time.time()
    res = dict(settings=dict(kit="circuitkit 0.2.4", task="multilingual_refusal", algorithm="eap-ig", level="node", n_pairs=N_PAIRS,
                             openers=OPENERS, ablation="mean over non-pad positions of the form's 60 dev prompts", random_seeds=5))
    try:
        attr, overlap = attribution()
        res.update(attribution=attr, overlap=overlap)
        json.dump(res, open(OUT / "step3.json", "w"), indent=1)
        res["causal"] = causal(attr)
        res["status"] = f"done in {time.time() - t0:.0f}s"
    except Exception as e:
        res["status"] = f"FAILED {type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-2500:]
    json.dump(res, open(OUT / "step3.json", "w"), indent=1, default=float)
    log("STEP3 " + res["status"])


if __name__ == "__main__":
    main()
