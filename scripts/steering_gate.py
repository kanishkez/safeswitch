"""Probe-gated activation steering. PRIMARY path: SafeTune CASTModel (`make_cast`). The hand-written
`GatedSteering` hook class below is a fallback only, used (and logged) if SafeTune fails.

Original fallback description: probe-gated activation steering hooks for Tiny Aya (cohere2), with or without a PEFT adapter.

Gate: at prefill, a logistic probe on the last-token residual at layer `gate_layer` gives p(harmful) per sequence.
If p > threshold (or the gate is off: threshold=None), add alpha * direction to the residual stream output of every
layer in `steer_layers`, at every position, for the rest of generation. Rows below threshold are unchanged.
"""
import re

import torch


def decoder_layers(model):
    layers = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)$", name)
        if m and "mlp" not in name and "attn" not in name:
            layers[int(m.group(1))] = mod
    assert layers, "no decoder layers found"
    return [layers[i] for i in sorted(layers)]


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class GatedSteering:
    def __init__(self, model, cfg):
        """cfg: dict(gate_layer, steer_layers, alpha, threshold or None, probe=dict(mean, std, coef, intercept), direction)"""
        self.cfg, self.layers, self.handles = cfg, decoder_layers(model), []
        dev = next(model.parameters()).device
        P = cfg["probe"]
        self.mu = torch.tensor(P["mean"], device=dev, dtype=torch.float32)
        self.sd = torch.tensor(P["std"], device=dev, dtype=torch.float32)
        self.w = torch.tensor(P["coef"], device=dev, dtype=torch.float32)
        self.b = float(P["intercept"])
        self.dir = torch.tensor(cfg["direction"], device=dev, dtype=torch.float32)
        self.gate = None          # (batch,) bool for the current generate() call
        self.n_fired, self.n_seen = 0, 0

    def _gate_hook(self, mod, inp, out):
        h = _hs(out)
        if h.shape[1] > 1:  # prefill: decide once per sequence (left padding -> last position is the prompt end)
            x = (h[:, -1, :].float() - self.mu) / self.sd
            p = torch.sigmoid(x @ self.w + self.b)
            thr = self.cfg["threshold"]
            self.gate = torch.ones_like(p, dtype=torch.bool) if thr is None else p > thr
            self.n_fired += int(self.gate.sum())
            self.n_seen += int(self.gate.numel())
        return out

    def _steer_hook(self, mod, inp, out):
        if self.gate is None or not bool(self.gate.any()):
            return out
        h = _hs(out)
        add = (self.cfg["alpha"] * self.dir).to(h.dtype)[None, None, :] * self.gate.to(h.dtype)[:, None, None]
        return _replace(out, h + add)

    def attach(self):
        self.handles.append(self.layers[self.cfg["gate_layer"]].register_forward_hook(self._gate_hook))
        for L in self.cfg["steer_layers"]:
            self.handles.append(self.layers[L].register_forward_hook(self._steer_hook))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ------------------------------------------------------------------ SafeTune path (primary)
class CountingCAST:
    """SafeTune CASTModel with the legacy logistic-probe gate (our probe, standardisation folded into the weights),
    per-prompt gating at prefill, steering vector added at the steering layer(s). Counts gate firings.
    `gated=False` -> SafeTune's unconditional install (ungated steering at the same alpha)."""

    def __init__(self, model, cfg, gated=True):
        import numpy as np
        from safetune.interventions.steer.cast import CASTModel
        P = cfg["probe"]
        mu, sd, coef = (np.asarray(P[k], dtype=np.float64) for k in ("mean", "std", "coef"))
        w = coef / sd
        b = float(P["intercept"]) - float((coef * mu / sd).sum())
        vecs = {int(L): torch.tensor(cfg["direction"], dtype=torch.float32) for L in cfg["steer_layers"]}
        self.cast = CASTModel(model, vecs, alpha=float(cfg["alpha"]), probe_layer=int(cfg["gate_layer"]),
                              probe_weights=torch.tensor(w, dtype=torch.float32), probe_bias=b,
                              threshold=float(cfg["threshold"]) if cfg.get("threshold") is not None else 0.5, per_prompt_gate=True)
        self.model, self.gated, self.n_fired, self.n_seen = model, gated, 0, 0
        orig = self.cast._gate_fires

        def counted(*a, **k):
            fires, score = orig(*a, **k)
            fl = fires if isinstance(fires, list) else [fires]
            self.n_fired += sum(bool(f) for f in fl)
            self.n_seen += len(fl)
            return fires, score
        self.cast._gate_fires = counted
        if not gated:
            self.cast.install()

    # generation API used by lora_v3_common.generate (model.generate(**enc, ...)) and attribute passthrough
    def generate(self, *a, **k):
        return self.cast.generate(*a, **k) if self.gated else self.model.generate(*a, **k)

    def __call__(self, *a, **k):  # logits for MC capability: gate on the prompt, then the steered forward pass
        if not self.gated:
            return self.model(*a, **k)
        ids = k.pop("input_ids", a[0] if a else None)
        return self.cast.forward(ids, **k)

    def __getattr__(self, name):
        return getattr(self.__dict__["model"], name)

    def detach(self):
        try:
            self.cast.remove()
        except Exception:
            self.cast._remove_steering()


def make_steered(model, cfg, gated=True):
    """Return (wrapped_model, backend). Tries SafeTune first; falls back to the hand hook (logged by the caller)."""
    try:
        return CountingCAST(model, cfg, gated=gated), "safetune.CASTModel(legacy logistic-probe gate)"
    except Exception as e:  # pragma: no cover
        gs = GatedSteering(model, dict(cfg, threshold=cfg.get("threshold") if gated else None)).attach()
        gs.model, gs.generate = model, model.generate
        return gs, f"fallback hand hook ({type(e).__name__}: {str(e)[:120]})"


# ------------------------------------------------------------------ attention type per layer (cohere2: 3 sliding-window : 1 global)
def layer_attention_types(config, n_layers):
    """List of 'global' / 'sliding_window' per decoder layer, read from the HF config (layer_types, else
    sliding_window_pattern: layer L is global iff (L + 1) % pattern == 0; Tiny Aya: 3, 7, ..., 35)."""
    lt = getattr(config, "layer_types", None)
    if lt and len(lt) == n_layers:
        return ["global" if ("full" in str(t) or "global" in str(t)) else "sliding_window" for t in lt]
    pat = getattr(config, "sliding_window_pattern", None) or 4
    return ["global" if (L + 1) % pat == 0 else "sliding_window" for L in range(n_layers)]


def nearest_global(types, target, lo, hi):
    """Nearest global layer to `target` within (lo, hi]; ties -> the deeper layer. None if there is none."""
    g = [L for L in range(lo + 1, hi + 1) if types[L] == "global"]
    return min(g, key=lambda L: (abs(L - target), -L)) if g else None
