#!/usr/bin/env python3
"""CircuitKIT refusal attribution on Tiny Aya Fire (PREREG_v4.md section 6a). Timeboxed; prints IDs/counts only.

Task: CircuitKIT's built-in `multilingual_refusal` (minimal harmful/harmless pairs, chat template ON, metric = logit
difference between the first token of a refusal opener and of a compliance opener). Algorithm EAP-IG, node level
(attention heads + MLPs), 20 pairs per language.
  English: the pairs bundled with CircuitKIT (binary_align/jailbreak_binary.csv), openers ("I", "Sure").
  Romanized Hindi / Hinglish: the same English pairs, each pair translated TOGETHER by the Sarvam-M translator (same
  tool and instructions as the test set) so the swap stays minimal; openers ("Main", "Zaroor") (CircuitKIT hi-Latn
  default) for Romanized and ("Sorry", "Sure") for Hinglish. Pairs that tokenise to different lengths are dropped by
  CircuitKIT; the kept count is logged.
Layer ranking: importance(layer) = sum over that layer's nodes of |attribution score| (English).
Pre-registered layer choice (used if the English run succeeds):
  gate/probe layer = most important layer in the middle third; steering layer = most important layer in
  (gate layer, n_layers - 2].
Writes results/probe_steering/circuitkit.json (+ the pairs CSVs). On failure, writes the exact error for the
"Feedback for kit maintainers" note and the hand-chosen layers stay.
"""
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_v3_common import MODEL_ID, MODEL_REV, ROOT, free, log  # noqa: E402  (applies the torchvision patch)

OUT = ROOT / "results/probe_steering"
N_PAIRS = 20
OPENERS = {"en": ("I", "Sure"), "Romanized Hindi": ("Main", "Zaroor"), "Hinglish": ("Sorry", "Sure")}


def layer_of(node):
    m = re.search(r"(\d+)", str(node))
    return int(m.group(1)) if m else None


def run_discovery(model, language, pairs_path, refuse, comply):
    import circuitkit as ck
    dp = dict(language="en" if language == "en" else "hi-Latn",
              num_examples=N_PAIRS, seed=0, data_partition="all", refuse=refuse, comply=comply)
    if pairs_path:
        dp["pairs_path"] = str(pairs_path)
    t0 = time.time()
    circ = ck.discover(model, "multilingual_refusal", algorithm="eap-ig", level="node", n_examples=N_PAIRS, batch_size=4,
                       sparsity=0.3, scope="both", output_path=str(OUT / f"circuitkit_{language.replace(' ', '_')}.pt"), data_params=dp)
    nodes = circ.top_nodes(10_000) if callable(getattr(circ, "top_nodes", None)) else []
    scored = []
    for n in nodes:
        if isinstance(n, (tuple, list)) and len(n) >= 2:
            scored.append((str(n[0]), float(n[1])))
        elif isinstance(n, dict):
            scored.append((str(n.get("node") or n.get("name")), float(n.get("score", 0.0))))
        else:
            scored.append((str(n), 1.0))
    imp = defaultdict(float)
    for name, s in scored:
        L = layer_of(name)
        if L is not None:
            imp[L] += abs(s)
    return dict(seconds=round(time.time() - t0, 1), data_params={k: v for k, v in dp.items() if k != "pairs_path"},
                n_nodes=len(scored), top_nodes=scored[:30], layer_importance=dict(sorted(imp.items())))


def load_tl(ck, res):
    try:
        return ck.load_model(MODEL_ID, revision=MODEL_REV, dtype="bfloat16", device="cuda")
    except TypeError:
        res["note_revision"] = "load_model did not accept revision; loaded the cached (pinned) snapshot"
        return ck.load_model(MODEL_ID, dtype="bfloat16", device="cuda")


def scores_from_sidecar(language):
    """Real per-node EAP-IG scores from CircuitKIT's `<output>_scores.pt` side-car (Circuit.top_nodes() returned 1.0 for every
    node in 0.2.4, so it carries rank only)."""
    import torch
    stem = "en" if language == "en" else language.replace(" ", "_")
    p = OUT / f"circuitkit_{stem}_scores.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    ns = {str(k): float(v) for k, v in d["node_scores"].items()}
    imp = defaultdict(float)
    for name, v in ns.items():
        L = layer_of(name)
        if L is not None:
            imp[L] += abs(v)
    top = sorted(ns.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return dict(score_file=p.name, n_nodes=len(ns), top_nodes=[(k, round(v, 6)) for k, v in top[:30]], layer_importance=dict(sorted(imp.items())))


def rescore():
    """Recompute layer importance, layer choice and overlap from the score side-cars; overwrite circuitkit.json."""
    res = json.load(open(OUT / "circuitkit.json"))
    n = res["model_cfg"]["n_layers"]
    for run, lang in [("English", "en"), ("Romanized Hindi", "Romanized Hindi"), ("Hinglish", "Hinglish")]:
        try:
            r = scores_from_sidecar(lang)
            res["runs"].setdefault(run, {})
            res["runs"][run]["top_nodes_api_note"] = "Circuit.top_nodes() returned score 1.0 for every node; scores below are from the side-car"
            res["runs"][run].update(r)
        except FileNotFoundError:
            pass
    imp = {int(k): v for k, v in res["runs"]["English"]["layer_importance"].items()}
    types = ["global" if (L + 1) % 4 == 0 else "sliding_window" for L in range(n)]  # matches TL cfg.attn_types for Tiny Aya
    mid = [L for L in range(n // 3, 2 * n // 3) if L in imp]
    gate = max(mid, key=imp.get)
    late = [L for L in imp if gate < L <= n - 2]
    steer = max(late, key=imp.get)
    rank = sorted(imp, key=imp.get, reverse=True)
    gl = [L for L in range(n) if types[L] == "global"]
    res["layer_choice"] = dict(gate_layer=gate, gate_attention=types[gate], steer_layer=steer, steer_attention=types[steer], global_layers=gl,
                               importance_rank_of_global_layers={L: rank.index(L) + 1 for L in gl if L in imp},
                               mean_importance=dict(global_=sum(imp[L] for L in gl if L in imp) / len(gl),
                                                    sliding=sum(v for L, v in imp.items() if L not in gl) / (len(imp) - len(gl))),
                               top10_layers=rank[:10], source="EAP-IG node_scores side-car (|score| summed per layer)",
                               rule="gate = most important middle-third layer; steer = most important layer in (gate, n-2]")
    def top_layers(r, k=5):
        li = {int(a): b for a, b in r["layer_importance"].items()}
        return sorted(li, key=li.get, reverse=True)[:k]
    def top_n(r, k=20):
        return {x for x, _ in r["top_nodes"][:k]}
    E = res["runs"]["English"]
    res["overlap"] = {f: dict(top5_layers_en=top_layers(E), top5_layers=top_layers(res["runs"][f]),
                              top5_layer_overlap=len(set(top_layers(E)) & set(top_layers(res["runs"][f]))),
                              top20_node_jaccard=round(len(top_n(E) & top_n(res["runs"][f])) / max(1, len(top_n(E) | top_n(res["runs"][f]))), 3))
                      for f in ["Romanized Hindi", "Hinglish"] if "layer_importance" in res["runs"].get(f, {}) and "score_file" in res["runs"][f]}
    if res.get("status") in (None, "OK", "ENGLISH_OK_LATIN_FAILED"):
        res["status"] = "OK" if len(res["overlap"]) == 2 else "ENGLISH_OK_LATIN_FAILED"
    res["fixes"] = ["layer ranking recomputed from node_scores side-car after top_nodes() returned uniform 1.0 (first pass gave a tie artifact: 12/13)"]
    json.dump(res, open(OUT / "circuitkit.json", "w"), indent=1, default=str)
    log(f"CIRCUITKIT RESCORED {res['status']} {res['layer_choice']} overlap={res['overlap']}")


def translate_pairs(en_pairs):
    import v4_build_fresh_test as b
    llm = b.LLM(b.TRANSLATOR)
    out = {}
    for form in ["Romanized Hindi", "Hinglish"]:
        convs = [[{"role": "system", "content": b.SYS_TRANSLATE},
                  {"role": "user", "content": f"{b.FORM_INSTR[form]}\nTranslate BOTH lines below. They differ in only one or two words; keep every other "
                                              f"word identical between the two translations. Output exactly two lines: the first translation, then the second.\n\n"
                                              f"{h}\n{s}"}] for h, s in en_pairs]
        res = llm.chat(convs, max_new_tokens=200)
        pairs = []
        for r in res:
            ls = [x.strip() for x in r.splitlines() if x.strip()]
            if len(ls) >= 2 and ls[0] != ls[1]:
                pairs.append((ls[0], ls[1]))
        out[form] = pairs
    free(llm)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(kit="circuitkit", version=None, task="multilingual_refusal", algorithm="eap-ig", level="node", n_pairs=N_PAIRS,
               openers=OPENERS, status=None, runs={})
    try:
        import circuitkit as ck
        res["version"] = getattr(ck, "__version__", None) or __import__("importlib.metadata").metadata.version("circuitkit")
        from circuitkit.tasks.builtins import multilingual_refusal as M
        en_all = M.load_pairs({"language": "en"})
        en = en_all[:60]
        model = load_tl(ck, res)
        res["model_cfg"] = dict(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, architecture=getattr(model.cfg, "original_architecture", None))
        log(f"CircuitKIT loaded {MODEL_ID}: {res['model_cfg']}")
        res["runs"]["English"] = run_discovery(model, "en", None, *OPENERS["en"])
        log(f"English discovery done in {res['runs']['English']['seconds']}s")
    except Exception as e:
        res["status"] = "FAILED"
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        json.dump(res, open(OUT / "circuitkit.json", "w"), indent=1, default=str)
        log(f"CIRCUITKIT FAILED {res['error'][:300]}")
        return
    n = res["model_cfg"]["n_layers"]
    imp = {int(k): v for k, v in res["runs"]["English"]["layer_importance"].items()}
    mid = [L for L in range(n // 3, 2 * n // 3) if L in imp]
    gate = max(mid, key=imp.get) if mid else None
    late = [L for L in imp if gate is not None and gate < L <= n - 2]
    steer = max(late, key=imp.get) if late else None
    at = [str(x) for x in (getattr(model.cfg, "attn_types", None) or [])]
    typ = lambda L: (("global" if at[L] == "global" else "sliding_window") if L is not None and len(at) == n else None)  # noqa: E731
    gl = [L for L in range(n) if typ(L) == "global"]
    res["layer_choice"] = dict(gate_layer=gate, gate_attention=typ(gate), steer_layer=steer, steer_attention=typ(steer), global_layers=gl,
                               importance_rank_of_global_layers={L: sorted(imp, key=imp.get, reverse=True).index(L) + 1 for L in gl if L in imp},
                               mean_importance=dict(global_=sum(imp.get(L, 0) for L in gl) / max(1, len(gl)),
                                                    sliding=sum(v for L, v in imp.items() if L not in gl) / max(1, len(imp) - len(gl))),
                               rule="gate = most important middle-third layer; steer = most important layer in (gate, n-2]")
    json.dump(res, open(OUT / "circuitkit.json", "w"), indent=1, default=str)
    # Latin-script runs
    try:
        free(model)
        tr = translate_pairs(en[: N_PAIRS * 2])
        import circuitkit as ck
        model = load_tl(ck, res)
        for form, pairs in tr.items():
            p = OUT / f"circuitkit_pairs_{form.replace(' ', '_')}.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["harmful", "harmless"])
                w.writerows(pairs)
            res["runs"][form] = run_discovery(model, form, p, *OPENERS[form])
            res["runs"][form]["pairs_translated"] = len(pairs)
            log(f"{form} discovery done")
        def top_layers(r, k=5):
            li = {int(a): b for a, b in r["layer_importance"].items()}
            return sorted(li, key=li.get, reverse=True)[:k]
        def top_n(r, k=20):
            return {x for x, _ in r["top_nodes"][:k]}
        E = res["runs"]["English"]
        res["overlap"] = {f: dict(top5_layers_en=top_layers(E), top5_layers=top_layers(res["runs"][f]),
                                  top5_layer_overlap=len(set(top_layers(E)) & set(top_layers(res["runs"][f]))),
                                  top20_node_jaccard=round(len(top_n(E) & top_n(res["runs"][f])) / max(1, len(top_n(E) | top_n(res["runs"][f]))), 3))
                          for f in ["Romanized Hindi", "Hinglish"] if f in res["runs"]}
        res["status"] = "OK"
    except Exception as e:
        res["status"] = "ENGLISH_OK_LATIN_FAILED"
        res["latin_error"] = f"{type(e).__name__}: {e}"
        res["latin_traceback"] = traceback.format_exc()[-3000:]
    json.dump(res, open(OUT / "circuitkit.json", "w"), indent=1, default=str)
    log(f"CIRCUITKIT {res['status']} layers={res['layer_choice']}")


if __name__ == "__main__":
    rescore() if "--rescore" in sys.argv else main()
