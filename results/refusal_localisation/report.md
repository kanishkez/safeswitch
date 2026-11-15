# Where refusal lives in Tiny Aya Fire (EXPLORATORY)

**Not pre-registered.** Dev prompts only: benchmarks v4 and v5, 60 harmful scenarios × 4 forms, plus 60 benign scenarios. The fresh test set was not used. Script: `scripts/refusal_localisation.py`, `scripts/refusal_components.py`; data `data/refusal_localisation/dev.jsonl` (sha `d1951390…`). Chat template ON. No headline table was changed, and no ablated weights were saved. Outputs are IDs and labels only.

**Metric (a): refusal logit-diff at the first response token.** This is logsumexp over the refusal-token set minus logsumexp over the compliance-token set.
- The sets were defined once, from the first tokens of greedy base replies to the 240 dev prompts (65 refusals, 175 compliances).
- Refusal set: "I'm", "म".
- Compliance set: 15 tokens, including newline, "Main", "मैं" and "It".

**Metric (b): generated refusal,** greedy, 128 tokens, judged by the 3-judge panel, on 20 scenarios per form.

**CIs:** scenario bootstrap, 2,000 resamples.

**Base logit-diff by form:**

| form | logit-diff [95% CI] |
|---|---|
| English | +1.43 [1.11, 1.71] |
| Hindi | −2.69 [−2.92, −2.47] |
| Hinglish | −3.12 [−3.55, −2.71] |
| Romanized Hindi | −4.88 [−5.32, −4.43] |

These are in the same order as actual refusal rates.

## Answers

**Which layers are necessary (step 1).**
- **Method:** project the English refusal direction (harmful − benign mean, last token, 2-fold cross-fitted) out of a single layer's output.
- **Layers 0–12:** no effect (|Δ| ≤ 0.23).
- **Layers 14–35:** every single layer lowers the English logit-diff by 1.2–2.25. That takes it from +1.43 to below zero at most layers.
- **Random directions:** 5 seeds, |Δ| < 0.18 at every layer.
- **Largest effects:** layers 35 (global, Δ 2.25), 23 (global, Δ 2.14) and 24 (Δ 2.03).
- **Cumulative ablation:** layers 0–12 together have no effect (Δ 0.30). Adding up to layer 16 gives Δ 2.22.
- **All layers at once:** Δ 2.98, against a random-direction Δ of 0.04 ± 0.24 (5 seeds). Generated English refusal falls from 80% [60, 95] to 15% [0, 35]; with a random direction it stays at 85%. Ablating layer 35 alone gives 60%.
- **Attention type:** global layers (3, 7, …, 35) are not systematically more necessary than their sliding-window neighbours.
- **Answer:** the English refusal direction becomes necessary from about layer 13–14 onward, and it is necessary at every layer after that.

**Where the script gap opens (step 2).**
- **Method:** patch the last-token residual of the English version of the same scenario into the Romanized or Hinglish prompt, one layer at a time. Recovery = share of the English − source logit-diff gap closed. The gap is 6.3 for Romanized and 4.6 for Hinglish.
- **Layers 0–2:** nothing is recovered.
- **Layers 3–13:** 10–29% is recovered. But an English *benign* donor recovers the same amount, so this early recovery is about language and format, not harm.
- **From layer 14:** the harmful-specific part (same-scenario minus benign donor) appears and grows.
  - Romanized recovery is 0.33 at layer 14 and 0.62 at layers 20–21. The benign donor gives 0.03–0.18.
  - Hinglish recovery is 0.36 at layer 14 and 0.56 at layer 20. The benign donor gives 0.13–0.18 at layers 14–15 and about 0 at layers 17–21.
- **Layer 22:** recovery jumps to 0.92–0.94, and is about 1.0 by layer 30. Late-layer recovery near 1 is partly trivial, because patching the last position there largely sets the next token directly.
- **Other harmful scenario as donor:** this recovers as much as the same scenario at every layer (for example 0.63 vs 0.62 at layer 20). The patched signal is a generic "this is harmful, refuse" state, not knowledge of the specific request.
- **Generation, patch at layer 35:**
  - Romanized refusal: 5% → 85% with the same-scenario donor, 85% with another harmful scenario, 5% with a benign donor.
  - Hinglish refusal: 20% → 75% (same scenario), 80% (other harmful), 20% (benign).
- **Answer:** the Latin-script prompts lack the generic harmful→refuse state that English builds between layers 14 and 22. Supplying it there restores refusal regardless of which harmful request it came from.

**Which components (step 3).**
- **Method:** CircuitKIT 0.2.4 EAP-IG, node level, on 20 minimal pairs per form. The openers are "I'm" vs newline, the dominant tokens of our sets. Romanized and Hinglish pairs were translated by Sarvam-M, 40 kept per form.
- **Top-20 sets:** mostly MLPs. Three heads appear: A14.13, A17.8 and A10.9.
- **Overlap with English:** Hinglish shares 14/20 (Jaccard 0.54), Romanized 11/20 (0.38). Romanized and Hinglish share 13/20 (0.48).
- **Shared by all three forms:** MLP 22, MLP 30, attention head **A14.13** (top 4 in every form), MLP 23, MLP 27 and others.
- **Causal check:** mean-ablate the form's own top-k components, measured on our dev metric.

  | form | top-5 drop | top-10 drop | top-20 drop | random components (same layers, 5 seeds) |
  |---|---|---|---|---|
  | English | 9.4 [9.0, 9.8] | 10.1 | 7.5 | ≤ 0.13 ± 0.29 |
  | Romanized | 0.3 | 0.9 | 1.1 [0.6, 1.6] | −0.3 |
  | Hinglish | 0.0 | −2.1 | 0.9 | ≈ 0 |

  Negative values mean refusal increased.

- **Latin forms:** these already sit far below zero, so there is little refusal to remove (floor effect).
- **English:** the top-5 set includes MLP 0 and MLP 35, and mean-ablating those may damage the model generally. We did not run a benign-side control, so this effect is not shown to be refusal-specific.
- **Answer:** attention head A14.13 and the MLPs around layers 20–31 (especially 22 and 30) are shared refusal components across forms. A14.13 sits at the layer where steps 1 and 2 find refusal becoming necessary and patchable.

**Where v4 acts (step 4).** **Not run.**
- The evaluated v4 adapter weights were only on a compute sandbox that expired before they could be downloaded, so the LoRA delta norms and band tests could not be computed.
- A re-trained v4, same pipeline and seed, is in progress. If its training data reproduces the original exactly, step 4 can be run on it and labelled as such.

## What this does not show

- **Not pre-registered, small dev set.** 60 harmful scenarios from older benchmarks, whose Romanized forms are machine-transliterated and often unnatural. The fresh test set was deliberately not used.
- **The refusal-token metric is a proxy.**
  - It has only 2 refusal tokens ("I'm" and the Devanagari "म").
  - Some compliance tokens ("Main", "मैं") also begin Hindi refusals, so the logit-diff understates Latin-script and Hindi refusal.
  - Generated, panel-judged refusal (metric b) was checked only for the top findings, on 20 scenarios.
- **Directional ablation shows necessity of one linear direction, not a mechanism.** Single-layer necessity from layer 14 onward is consistent with a direction that is written once and read many times, so the effects are not additive across layers.
- **Patching recovery near 1 at late layers is partly trivial.** At late layers the patched last-token state largely determines the next token. The informative range is layers 14–22.
- **No scenario-specific signal.** The "other harmful scenario" donor shows the patched signal is generic. It does not show where request-specific harm understanding lives.
- **Component ablation (step 3) has no benign control.** The English top-k drop may partly be general damage from ablating MLP 0 and MLP 35. Latin-script ablation is limited by a floor effect. CircuitKIT attribution used single opener tokens and 20 minimal pairs per form.
- **Nothing here concerns the fine-tuned models.** Step 4 was not run, so we cannot say where v3 or v4 act.
