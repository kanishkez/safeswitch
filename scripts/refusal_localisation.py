#!/usr/bin/env python3
"""EXPLORATORY: localise refusal in Tiny Aya Fire (not pre-registered; dev prompts only, never the fresh test set).

Data: data/refusal_localisation/dev.jsonl (benchmarks v4/v5: 60 harmful + 60 benign scenarios x 4 forms; extra English
fit prompts from benchmark v6). All text-disjoint from data/v4_fresh/test.jsonl. Chat template ON. IDs only in outputs.

Metric (a): refusal logit-diff at the first response position = logsumexp(logits[R]) - logsumexp(logits[C]), R/C = first-token
sets of base refusals / base compliances (lexical refusal on greedy base replies to the 240 eval prompts), defined once.
Metric (b): generated refusal (greedy, 128 tokens), panel-judged, 20 scenarios/form, for the top findings only.
CIs: scenario bootstrap (2,000).

Step 1  necessity by layer: project the English refusal direction (diff of means harmful-benign, last token, per layer,
        2-fold cross-fitted over scenarios) out of the residual output of one layer (all positions), or of every layer.
        Control: random unit direction (5 seeds) -- projection removes one direction regardless of norm.
Step 2  activation patching: Hinglish / Romanized prompt, replace the last-token residual output of layer L with the
        English same-scenario value; donors: English same scenario, English benign (control), English other harmful
        scenario (control). Recovery = (patched - source) / (English - source).
Step 3  components (CircuitKIT): only if time remains.
Step 4  v4: LoRA delta norms per layer/module; band variants. Runs only if the v4 adapter is present.
Writes results/refusal_localisation/{summary.json, report.md} (+ step jsons). No model weights are saved.
"""
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_v3_common import ROOT, chat_prompt, free, judge_all, lexical_refusal, load_generator, log, panel, read_jsonl  # noqa: E402
from steering_gate import decoder_layers  # noqa: E402

OUT = ROOT / "results/refusal_localisation"
DEV = ROOT / "data/refusal_localisation/dev.jsonl"
V4 = ROOT / "results/lora_v4/adapter/checkpoint-150"
FORMS = ["English", "Hindi", "Romanized Hindi", "Hinglish"]
LATIN = ["Romanized Hindi", "Hinglish"]
SEEDS = 5
B = 2000
TIME_LIMIT_UTC = (16, 25)  # 21:55 IST: stop starting new steps


def time_left():
    t = time.gmtime()
    return (t.tm_hour, t.tm_min) < TIME_LIMIT_UTC


def gtype(L):
    return "global" if (L + 1) % 4 == 0 else "sliding"


def boot_mean(vals_by_scen, seed=0):
    keys = sorted(vals_by_scen)
    v = np.array([np.mean(vals_by_scen[k]) for k in keys])
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(v))


def boot_ratio(num_by, den_by, seed=0):
    """ratio of means sum(num)/sum(den) with scenario bootstrap."""
    keys = sorted(num_by)
    n = np.array([num_by[k] for k in keys])
    d = np.array([den_by[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(keys), len(keys))
        bs.append(n[i].mean() / d[i].mean() if abs(d[i].mean()) > 1e-6 else np.nan)
    return dict(mean=float(n.mean() / d.mean()), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=len(keys))


class Runner:
    def __init__(self, tok, model):
        self.tok, self.model, self.layers = tok, model, decoder_layers(model)
        self.n = len(self.layers)
        self.tok.padding_side = "left"
        self.R = self.C = None

    def enc(self, items):
        texts = [chat_prompt(self.tok, it) for it in items]
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)

    @torch.no_grad()
    def resid_last(self, items, bs=16):
        """(N, n_layers, d) last-token residual output of every decoder layer (float32, CPU)."""
        out = []
        for i in range(0, len(items), bs):
            cap = {}
            hs = [L.register_forward_hook(lambda m, a, o, j=j: cap.__setitem__(j, (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
                  for j, L in enumerate(self.layers)]
            try:
                self.model(**self.enc(items[i:i + bs]))
            finally:
                for h in hs:
                    h.remove()
            out.append(torch.stack([cap[j] for j in range(self.n)], 1))
        return torch.cat(out)

    @torch.no_grad()
    def logits_last(self, items, hooks_fn=None, bs=16):
        res = []
        for i in range(0, len(items), bs):
            hs = hooks_fn(i, min(i + bs, len(items))) if hooks_fn else []
            try:
                lg = self.model(**self.enc(items[i:i + bs])).logits[:, -1].float()
            finally:
                for h in hs:
                    h.remove()
            res.append(lg.cpu())
        return torch.cat(res)

    def ld(self, lg):
        return (torch.logsumexp(lg[:, self.R], -1) - torch.logsumexp(lg[:, self.C], -1)).numpy()

    @torch.no_grad()
    def generate(self, items, hooks_fn=None, bs=16, max_new=128):
        outs = []
        for i in range(0, len(items), bs):
            enc = self.enc(items[i:i + bs])
            hs = hooks_fn(i, min(i + bs, len(items))) if hooks_fn else []
            try:
                g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05, pad_token_id=self.tok.pad_token_id)
            finally:
                for h in hs:
                    h.remove()
            outs += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    # ---- hooks
    def ablate_hooks(self, dirs):
        """dirs: {layer: unit vector (d,) or (N,d) per-row}. Projects the direction out of the layer output, all positions."""
        def mk(u):
            def h(m, a, o):
                x = o[0] if isinstance(o, tuple) else o
                uu = u.to(x.device, x.dtype)
                if uu.dim() == 1:
                    x = x - (x @ uu)[..., None] * uu
                else:
                    x = x - (x * uu[:, None, :]).sum(-1, keepdim=True) * uu[:, None, :]
                return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            return h
        return lambda i, j: [self.layers[L].register_forward_hook(mk(u if u.dim() == 1 else u[i:j])) for L, u in dirs.items()]

    def patch_hooks(self, L, donor):
        """replace last-token residual output of layer L (prefill only) with donor[i:j] (N, d)."""
        def mk(dn):
            def h(m, a, o):
                x = o[0] if isinstance(o, tuple) else o
                if x.shape[1] > 1:
                    x = x.clone()
                    x[:, -1] = dn.to(x.device, x.dtype)
                    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
                return o
            return h
        return lambda i, j: [self.layers[L].register_forward_hook(mk(donor[i:j]))]


def define_token_sets(run, ev):
    outs = run.generate(ev, max_new=24)
    ids_r, ids_c = Counter(), Counter()
    for it, o in zip(ev, outs):
        t = run.tok(o, add_special_tokens=False)["input_ids"]
        if not t:
            continue
        (ids_r if lexical_refusal(o) else ids_c)[t[0]] += 1
    R = [t for t, c in ids_r.items() if c >= 2 and ids_c[t] < c / 3]
    C = [t for t, c in ids_c.items() if c >= 2 and ids_r[t] < c / 3]
    return R, C, dict(n_refusals=sum(ids_r.values()), n_compliance=sum(ids_c.values()),
                      refusal_tokens={str(t): [run.tok.decode([t]), ids_r[t]] for t in R},
                      compliance_tokens={str(t): [run.tok.decode([t]), ids_c[t]] for t in C})


def judge_gen(rows):
    judge_all(rows)
    for r in rows:
        r["panel_refusal"], r["panel_harm"] = panel(r)
    return rows


def gen_refusal(arm_rows):
    by = defaultdict(list)
    for r in arm_rows:
        by[r["scenario_id"]].append(float(bool(r["panel_refusal"])))
    return boot_mean(by)


# --------------------------------------------------------------------------------------- steps
def step1(run, ev, fitH, fitB, S):
    en = [r for r in ev if r["language"] == "English"]
    scen = sorted({r["scenario_id"] for r in en})
    rng = random.Random(1)
    rng.shuffle(scen)
    fold = {s: k % 2 for k, s in enumerate(scen)}
    benign_en = [r for r in S["benign"] if r["language"] == "English"]
    bscen = sorted({r["scenario_id"] for r in benign_en})
    bfold = {s: k % 2 for k, s in enumerate(bscen)}
    Hen = run.resid_last(en)
    Hb = run.resid_last(benign_en)
    HfH, HfB = run.resid_last(fitH), run.resid_last(fitB)
    n, d = run.n, Hen.shape[-1]
    # per-row direction (N, n, d): fitted on the other fold + extra fit prompts
    D = torch.zeros(len(en), n, d)
    for k in (0, 1):
        h = torch.cat([Hen[[i for i, r in enumerate(en) if fold[r["scenario_id"]] != k]], HfH])
        b = torch.cat([Hb[[i for i, r in enumerate(benign_en) if bfold[r["scenario_id"]] != k]], HfB])
        r_ = h.mean(0) - b.mean(0)
        r_ = r_ / r_.norm(dim=-1, keepdim=True)
        for i, r in enumerate(en):
            if fold[r["scenario_id"]] == k:
                D[i] = r_
    base = run.ld(run.logits_last(en))
    sid = [r["scenario_id"] for r in en]

    def summ(v):
        by = defaultdict(list)
        for s, x in zip(sid, v):
            by[s].append(x)
        return boot_mean(by)

    res = dict(base=summ(base), single={}, random_single={}, n_prompts=len(en))
    gen = torch.Generator().manual_seed(0)
    for L in range(n):
        v = run.ld(run.logits_last(en, run.ablate_hooks({L: D[:, L]})))
        res["single"][L] = dict(summ(v), drop=float(base.mean() - v.mean()), attention=gtype(L))
        rs = []
        for s in range(SEEDS):
            u = torch.randn(d, generator=torch.Generator().manual_seed(1000 * s + L))
            rs.append(float(base.mean() - run.ld(run.logits_last(en, run.ablate_hooks({L: u / u.norm()}))).mean()))
        res["random_single"][L] = dict(drop_mean=float(np.mean(rs)), drop_max=float(np.max(np.abs(rs))))
    v_all = run.ld(run.logits_last(en, run.ablate_hooks({L: D[:, L] for L in range(n)})))
    res["all_layers"] = dict(summ(v_all), drop=float(base.mean() - v_all.mean()))
    ra = []
    for s in range(SEEDS):
        us = {L: (lambda u: u / u.norm())(torch.randn(d, generator=torch.Generator().manual_seed(77 + 1000 * s + L))) for L in range(n)}
        ra.append(float(base.mean() - run.ld(run.logits_last(en, run.ablate_hooks(us))).mean()))
    res["random_all_layers"] = dict(drop_mean=float(np.mean(ra)), drop_sd=float(np.std(ra)), drops=ra)
    best = max(range(n), key=lambda L: res["single"][L]["drop"])
    res["best_single_layer"] = best
    # cumulative necessity: ablate layers 0..L (finds where removing becomes sufficient to kill refusal)
    res["prefix"] = {}
    for L in range(0, n, 4):
        v = run.ld(run.logits_last(en, run.ablate_hooks({k: D[:, k] for k in range(L + 1)})))
        res["prefix"][L] = float(base.mean() - v.mean())
    # (b) generation on 20 English scenarios: base, all-layer ablation, random all-layer (seed 0), best single layer
    g_items = [r for r in en if r["scenario_id"] in set(sorted(set(sid))[:20])]
    gi = [en.index(r) for r in g_items]
    Dg = D[gi]
    us0 = {L: (lambda u: u / u.norm())(torch.randn(d, generator=torch.Generator().manual_seed(77 + L))) for L in range(n)}
    arms = {"base": None, "ablate_all_layers": run.ablate_hooks({L: Dg[:, L] for L in range(n)}),
            "random_all_layers_seed0": run.ablate_hooks(us0), f"ablate_layer{best}": run.ablate_hooks({best: Dg[:, best]})}
    rows = []
    for a, hf in arms.items():
        for it, o in zip(g_items, run.generate(g_items, hf)):
            rows.append(dict(it, response=o, arm=a))
    return res, rows


def step2(run, ev, S):
    byf = {f: sorted([r for r in ev if r["language"] == f], key=lambda r: r["scenario_id"]) for f in FORMS}
    en = byf["English"]
    scen = [r["scenario_id"] for r in en]
    Hen = run.resid_last(en)
    benign_en = sorted([r for r in S["benign"] if r["language"] == "English"], key=lambda r: r["scenario_id"])[:len(en)]
    Hben = run.resid_last(benign_en)
    perm = scen[1:] + scen[:1]  # derangement: other harmful scenario
    Hother = Hen[[scen.index(s) for s in perm]]
    ld_en = run.ld(run.logits_last(en))
    res = dict(english=boot_mean({s: [x] for s, x in zip(scen, ld_en)}), forms={})
    for f in LATIN:
        src = byf[f]
        assert [r["scenario_id"] for r in src] == scen
        ld_src = run.ld(run.logits_last(src))
        fr = dict(source=boot_mean({s: [x] for s, x in zip(scen, ld_src)}), gap=float(ld_en.mean() - ld_src.mean()), layers={})
        for L in range(run.n):
            row = {}
            for name, donor in [("english_same", Hen[:, L]), ("english_benign", Hben[:, L]), ("english_other_harmful", Hother[:, L])]:
                v = run.ld(run.logits_last(src, run.patch_hooks(L, donor)))
                row[name] = dict(ld=float(v.mean()),
                                 recovery=boot_ratio({s: x - y for s, x, y in zip(scen, v, ld_src)}, {s: e - y for s, e, y in zip(scen, ld_en, ld_src)}))
            row["attention"] = gtype(L)
            fr["layers"][L] = row
        rec = {L: fr["layers"][L]["english_same"]["recovery"]["mean"] for L in range(run.n)}
        fr["first_layer_recovery_ge_0_5"] = next((L for L in range(run.n) if rec[L] >= 0.5), None)
        fr["last_layer_recovery_ge_0_5"] = max((L for L in range(run.n) if rec[L] >= 0.5), default=None)
        res["forms"][f] = fr
    # (b) generation: patch at the layer with the highest same-scenario recovery among the earliest half of layers it holds
    rows = []
    for f in LATIN:
        fr = res["forms"][f]
        Lp = fr["last_layer_recovery_ge_0_5"]
        Lp = Lp if Lp is not None else max(range(run.n), key=lambda L: fr["layers"][L]["english_same"]["recovery"]["mean"])
        fr["generation_patch_layer"] = Lp
        keep = scen[:20]
        src = [r for r in byf[f] if r["scenario_id"] in keep]
        idx = [scen.index(r["scenario_id"]) for r in src]
        for a, hf in {"source": None, f"patch_L{Lp}_english_same": run.patch_hooks(Lp, Hen[idx, Lp]),
                      f"patch_L{Lp}_english_benign": run.patch_hooks(Lp, Hben[idx, Lp]),
                      f"patch_L{Lp}_english_other": run.patch_hooks(Lp, Hother[idx, Lp])}.items():
            for it, o in zip(src, run.generate(src, hf)):
                rows.append(dict(it, response=o, arm=f"{f}|{a}"))
    return res, rows


def step4(S, ev):
    """v4: delta norms + band variants. Requires the v4 adapter."""
    from peft import PeftModel  # noqa: F401
    from safetensors.torch import load_file
    import re as _re
    sd = load_file(str(V4 / "adapter_model.safetensors"))
    cfg = json.load(open(V4 / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    norms = defaultdict(float)
    for k, A in sd.items():
        if "lora_A" not in k:
            continue
        Bm = sd[k.replace("lora_A", "lora_B")]
        L = int(_re.search(r"layers\.(\d+)\.", k).group(1))
        mod = k.split(".")[-3]
        norms[(L, mod)] = float(torch.linalg.matrix_norm((Bm.float() @ A.float()) * scale))
    by_layer = defaultdict(float)
    for (L, m), v in norms.items():
        by_layer[L] += v ** 2
    res = dict(delta_norm_by_layer={L: math.sqrt(v) for L, v in sorted(by_layer.items())},
               delta_norm_by_module={f"{L}.{m}": v for (L, m), v in sorted(norms.items())})
    tok, model = load_generator(V4)
    run = Runner(tok, model)
    run.R, run.C = S["R"], S["C"]
    lora = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    orig = {n: p.detach().clone() for n, p in lora.items()}
    bands = {"0-8": range(0, 9), "9-17": range(9, 18), "18-26": range(18, 27), "27-35": range(27, 36)}
    variants = {"full_v4": set(range(36))}
    for b, rg in bands.items():
        variants[f"only_{b}"] = set(rg)
        variants[f"without_{b}"] = set(range(36)) - set(rg)
    variants["no_adapter"] = set()
    lat = [r for r in ev if r["language"] in LATIN]
    g20 = sorted({r["scenario_id"] for r in lat})[:20]
    gl = [r for r in lat if r["scenario_id"] in g20]
    ben = [r for r in S["benign"] if r["language"] in LATIN and r["scenario_id"] in sorted({x["scenario_id"] for x in S["benign"]})[:20]]
    res["variants"], rows = {}, []
    for name, keep in variants.items():
        with torch.no_grad():
            for n_, p in lora.items():
                L = int(_re.search(r"layers\.(\d+)\.", n_).group(1))
                p.copy_(orig[n_] if L in keep else torch.zeros_like(p))
        v = run.ld(run.logits_last(lat))
        res["variants"][name] = {f: boot_mean({r["scenario_id"]: [x] for r, x in zip(lat, v) if r["language"] == f}) for f in LATIN}
        for it, o in zip(gl + ben, run.generate(gl + ben)):
            rows.append(dict(it, response=o, arm=name))
    with torch.no_grad():
        for n_, p in lora.items():
            p.copy_(orig[n_])
    free(model)
    return res, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = read_jsonl(DEV)
    ev = [r for r in dev if r["role"] == "eval_harmful"]
    S = dict(benign=[r for r in dev if r["role"] == "benign"])
    fitH = [r for r in dev if r["role"] == "fit_harmful"]
    fitB = [r for r in dev if r["role"] == "fit_benign"]
    summary = dict(status={}, data=dict(dev_sha=__import__("hashlib").sha256(open(DEV, "rb").read()).hexdigest(), eval_harmful_prompts=len(ev),
                                        eval_harmful_scenarios=len({r["scenario_id"] for r in ev}), fit_extra=dict(harmful=len(fitH), benign=len(fitB))))
    tok, model = load_generator(None)
    run = Runner(tok, model)
    R, C, info = define_token_sets(run, ev)
    run.R, run.C = R, C
    S["R"], S["C"] = R, C
    summary["token_sets"] = info
    base_ld = run.ld(run.logits_last(ev))
    summary["base_ld_by_form"] = {f: boot_mean({r["scenario_id"]: [x] for r, x in zip(ev, base_ld) if r["language"] == f}) for f in FORMS}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    gen_rows = []
    for name, fn in [("step1", lambda: step1(run, ev, fitH, fitB, S)), ("step2", lambda: step2(run, ev, S))]:
        if not time_left():
            summary["status"][name] = "not run (timebox)"
            continue
        t0 = time.time()
        try:
            res, rows = fn()
            summary[name] = res
            gen_rows += [dict(r, step=name) for r in rows]
            summary["status"][name] = f"done in {time.time() - t0:.0f}s"
        except Exception as e:
            import traceback
            summary["status"][name] = f"FAILED {type(e).__name__}: {e}"
            summary[name + "_traceback"] = traceback.format_exc()[-2000:]
        log(f"{name}: {summary['status'][name]}")
        json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    free(model)
    if (V4 / "adapter_model.safetensors").exists() and time_left():
        t0 = time.time()
        try:
            res, rows = step4(S, ev)
            summary["step4"] = res
            gen_rows += [dict(r, step="step4") for r in rows]
            summary["status"]["step4"] = f"done in {time.time() - t0:.0f}s"
        except Exception as e:
            import traceback
            summary["status"]["step4"] = f"FAILED {type(e).__name__}: {e}"
            summary["step4_traceback"] = traceback.format_exc()[-2000:]
    else:
        summary["status"]["step4"] = "not run: v4 adapter not available on this sandbox (only copy was on sandbox 1, offline)" if not (V4 / "adapter_model.safetensors").exists() else "not run (timebox)"
    summary["status"]["step3"] = "not run (timebox: step 3 only if time remains)"
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    if gen_rows:
        judge_gen(gen_rows)
        gsum = defaultdict(dict)
        for st in sorted({r["step"] for r in gen_rows}):
            for a in sorted({r["arm"] for r in gen_rows if r["step"] == st}):
                sub = [r for r in gen_rows if r["step"] == st and r["arm"] == a]
                harm = [r for r in sub if r["intent"] == "harmful"]
                ben = [r for r in sub if r["intent"] == "benign"]
                gsum[st][a] = dict(refusal=gen_refusal(harm) if harm else None,
                                   benign_helpful=boot_mean({r["scenario_id"]: [float(not r["panel_refusal"])] for r in ben}) if ben else None,
                                   n=len(sub))
        summary["generation"] = gsum
        with open(OUT / "gen_rows_ids.jsonl", "w") as fh:  # IDs + labels only, no text
            for r in gen_rows:
                fh.write(json.dumps({k: r.get(k) for k in ["step", "arm", "sample_id", "scenario_id", "language", "intent", "panel_refusal", "panel_harm"]}) + "\n")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    (OUT / "DONE").write_text("ok\n")
    log("LOCALISATION DONE " + json.dumps(summary["status"]))


if __name__ == "__main__":
    main()
