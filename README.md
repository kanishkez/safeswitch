# SwitchSafe-HI: Auditing and Fixing Code Mixed Hindi Safety in Tiny Aya Fire

**One-line claim:** A LoRA adapter trained mostly on Latin script Hindi raises Tiny Aya Fire's refusal of harmful requests from 43.9% to 79.6% on a fresh pre-registered test set, and from 21.4% to 76.2% in Romanized Hindi and Hinglish, without hurting helpfulness or MMLU. Inside the model, the gap traces back to a refusal state that English prompts build in layers 14 to 22 and Latin script prompts never reach.

**Tracks + problem statements covered:**
- Track 2a (AI Red Teaming, Safety & Alignment): Problem Statement 4, Multilingual & Code-Mixed Safety

## Summary

Hundreds of millions of people type Hindi in Latin script, either as Romanized Hindi or as Hinglish, which mixes Hindi and English. Tiny Aya Fire refuses 76% of harmful requests in English but only about 21% of the same requests written in Romanized Hindi or Hinglish. For those users, the script works as a free jailbreak.

I evaluated the model across English, Devanagari Hindi, Romanized Hindi and Hinglish on a test set built from scratch, with a three judge safety panel and a pre-registered analysis plan. A LoRA adapter, SafeSwitch LoRA, with 68% Latin script training data closed most of the gap. Both pre-registered hypotheses held: it beats the base model by +35.7 points [+29.7, +41.6], and it closes more of the Latin script gap than a LoRA baseline trained on less Latin script data, by +19.2 points [+13.0, +25.7]. MMLU is unchanged. A second training run with the same recipe reproduces the result on validation.

To understand why the gap exists, I traced refusal through the model. It recognises harm in Latin script Hindi (probe AUC 0.99) but does not convert that recognition into a refusal. Patching English activations into Romanized prompts closes 62% of the refusal gap by layer 20 and 92% by layer 22, and attention head 13 in layer 14 (A14.13) is one of the top refusal components in all three Latin and English forms.

## Abstract

See [`ABSTRACT.md`](https://github.com/kanishkez/safeswitch/blob/main/ABSTRACT.md) (under 500 words).

## Headline results

Fresh pre-registered test set ([`PREREG_v4.md`](https://github.com/kanishkez/safeswitch/blob/main/PREREG_v4.md), [full report](https://github.com/kanishkez/safeswitch/blob/main/results/v4_fresh/report.md)): 128 scenarios (52 harmful, 38 ordinary, 38 scary-but-benign), each in four forms, for 512 prompts. Every prompt received one greedy and three sampled replies with the chat template on, and the set was evaluated exactly once. Intervals are scenario-level bootstrap 95% CIs.

| arm | harmful refusal | refusal, Romanized + Hinglish | attack success per reply | broken within 4 tries | ordinary helpfulness | scary false refusal | loops |
|---|---|---|---|---|---|---|---|
| base | 43.9% | 21.4% | 23.8% | 43.8% | 91.5% | 4.3% | 6.9% |
| prompt scaffold | 59.9% | 45.4% | 13.5% | 33.7% | 92.4% | 10.9% | 5.7% |
| LoRA baseline | 67.9% | 57.0% | 12.7% | 26.4% | 98.5% | 8.4% | 1.5% |
| **SafeSwitch LoRA** | **79.6%** | **76.2%** | **8.6%** | **16.4%** | 96.9% | 10.7% | 1.7% |
| SafeSwitch LoRA + scaffold | 92.3% | 93.0% | 1.4% | 3.4% | 98.4% | 34.7% | 0.4% |
| base + gated steering | 55.6% | 26.4% | 26.8% | 49.5% | 91.5% | 4.3% | 15.5% |
| SafeSwitch LoRA + gated steering | 84.6% | 78.8% | 16.7% | 28.8% | 96.9% | 10.7% | 25.7% |

Paired differences for SafeSwitch LoRA against the base model: refusal +35.7 [+29.7, +41.6], Latin script refusal +54.8 [+45.2, +64.4], attack success −15.1 [−20.1, −10.6]. Against the LoRA baseline: refusal +11.7 [+8.2, +15.4], Latin script refusal +19.2 [+13.0, +25.7], attack success −4.1 [−6.9, −1.6].

**Base model + method in one line:** `CohereLabs/tiny-aya-fire` (3.35B) with SafeSwitch LoRA (r = 16 on all linear layers, step 150 of 200, chosen on validation), trained with completion-only loss on about 600 pairs, 68% of them Romanized Hindi or Hinglish. The pairs are in-language refusals and the base model's own helpful answers, filtered by a judge panel and by CuratorKIT.

## Problem statement and approach

Safety tuning and safety judges are built mostly around English. Hindi speakers, however, write in three ways: Devanagari, Romanized Hindi and Hinglish. I set out to answer two questions. Does Tiny Aya Fire refuse harmful requests equally in all four forms? And can a cheap intervention close any gap without making the model refuse harmless questions?

The question matters in practice. Small open multilingual models like Tiny Aya are the ones that get deployed on device and in low cost Indian language assistants, and their users write in Latin script every day.

The work proceeded in four stages: measuring baseline refusal per form with a multi-judge panel; hardening the evaluation pipeline until results stopped moving; comparing three interventions (a prompt scaffold, activation steering and LoRA fine-tuning) under a pre-registration; and locating where refusal breaks down inside the model.

**Threat model.** A regular user sends a direct harmful request in one of the four forms, without a jailbreak template. The only attacker effort modelled is asking again; cost-to-break is the number of attempts needed to get one harmful reply. Optimised jailbreaks, multi-turn attacks and white-box attacks are out of scope.

## Hackathon repos (kits) used

**CuratorKIT (0.1.22)** curated every new training pair ([manifest](https://github.com/kanishkez/safeswitch/blob/main/data/lora_v4/manifest.json)) through a schema gate, exact deduplication, MinHash near-deduplication (0.85), cleaning and LaBSE embedding deduplication (0.95). 163 of 165 helpful pairs and 432 of 525 refusals passed. It also built the curated seed manifest and deduplicated the earlier benchmarks.

**SafeTune (0.1.12)** provided the steering. The refusal direction comes from `extract_caa_vectors` (cosine 1.00 with my own difference of means), and `CASTModel` applies it in logistic-probe gate mode, with my English-trained probe as the gate.

**CircuitKIT (0.2.4)** provided circuit attribution. I ran its `multilingual_refusal` task with EAP-IG at node level on 20 minimal pairs per form, through its TransformerLens loader (checked against Hugging Face logits, max |Δ| = 1.4e-5). The English layer ranking chose the probe layer (16) and steering layer (20) before any steering output existed, and the per-form runs give the component overlap in Findings 4 and 5. Two notes for the maintainers: in 0.2.4, `Circuit.top_nodes()` returns 1.0 for every node, so the real scores have to be read from the `*_scores.pt` side-car file; and the memory estimate (about 216 GB at `seq=8192`) was far above actual use (about 20 GB).

**AuditKIT (1.1.3)** produced fingerprinted `RunResult` audit reports for the earlier steering and prompt-repair runs. Final judging uses my own judge adapters.

TRL 1.18 and PEFT 0.21 handle LoRA training. CuratorKIT feeds the training data, CircuitKIT picks the layers and maps the refusal components, and SafeTune applies the steering. Every arm is scored by the same judge panel and the same loop and preamble detectors.

## Models and licenses

- **Base Aya model:** `CohereLabs/tiny-aya-fire` at revision `73500c80fcb6c8bea3e830c21d5d2920249a0e1a` (3.35B, cohere2 architecture), loaded in bfloat16 without quantisation and always prompted through its chat template. License: CC-BY-NC 4.0 with the Cohere acceptable use policy.
- **Target / hardened model:** the base model with SafeSwitch LoRA, [`kanishkez/safeswitch-lora`](https://huggingface.co/kanishkez/safeswitch-lora), released under CC-BY-NC 4.0 to match the base model.
- **Judges:** `Qwen/Qwen3Guard-Gen-4B` (Apache 2.0), `meta-llama/Llama-Guard-3-8B` (Llama 3.1 Community License) and `ToxicityPrompts/PolyGuard-Qwen-Smol` (CC-BY 4.0).
- **Helper models:** `sarvamai/sarvam-m` for translation and `sentence-transformers/LaBSE` for overlap checks (both Apache 2.0).

## Datasets, benchmarks and licenses

The fresh test set draws on sources not used anywhere earlier in the project:
- **Harmful:** `CohereLabs/aya_redteaming` (Hindi, Apache 2.0), red-team prompts written by native speakers together with their human English translations, screened to keep requests that seek harmful help and balanced across harm categories.
- **Ordinary benign:** `CohereLabs/aya_dataset` (Hindi, Apache 2.0), native prompts screened as harmless and self-contained.
- **Scary-but-benign:** the 250 safe prompts in `Paul/XSTest` (CC-BY 4.0), balanced across its ten categories.

Missing forms were produced by Sarvam-M from the Hindi text and an English gloss, written the way people type in chat apps rather than transliterated letter by letter. Every item records its provenance per form.

Before any generation, each test prompt was checked against all 1,441 earlier prompts in the project on exact text, normalised text and LaBSE similarity (cosine ≥ 0.90), both before and after translation. There were zero overlaps on all three checks (highest cosine 0.838), and the same held against the training pool. The file was then committed with its SHA-256 (`b78080bf…`).

The training data adds new Romanized Hindi and Hinglish pairs to 373 pairs from the earlier baseline. Harmful prompts are paired with in-language refusals generated by the base model under a two-example scaffold; benign prompts are paired with the base model's own answers. Refusals must pass Qwen3Guard and a lexical check, and every pair passes loop, language, preamble and length filters before CuratorKIT. The final set has 618 pairs (265 refusals, 353 helpful, 67.6% Latin script). Validation uses separate scenarios from both training and test.

**Languages covered.** English, Hindi in Devanagari, Romanized Hindi and Hinglish, in both training and evaluation.

**Train / test splits and leak check.**

| split | scenarios | prompts | notes |
|---|---|---|---|
| training | new pool scenarios (disjoint from validation and test) plus 373 earlier pairs | 618 pairs (English 100, Hindi 100, Romanized 195, Hinglish 223) | 265 refusals, 353 helpful |
| validation | 74 | 296 | 88 harmful, 112 ordinary, 96 scary-but-benign prompts; checkpoint selection only |
| test | 128 | 512 | 52 harmful, 38 ordinary, 38 scary-but-benign scenarios × 4 forms; evaluated once |

Leak checks used exact match, normalised text and LaBSE embedding similarity (cosine ≥ 0.90) between the test set and every other prompt in the project, including the training pool. All three came back at zero.

**Safety benchmark.** The fresh test set is the safety benchmark and the focus of the work. It covers all four forms; the harmful and scary-but-benign items measure refusal and over-refusal respectively.

**General capability benchmarks.** MMLU (English) and Global-MMLU (Hindi), 300 items each, used as retention checks to confirm that safety tuning did not reduce general ability.

**CuratorKIT filtering.** Candidate pairs first went through my own filters: of 616 benign candidates, 165 survived (rejections were mostly unfinished, wrong-language, too-short or looping answers); of 1,360 refusal candidates, 525 survived (most rejections were outputs that Qwen3Guard and the lexical matcher did not judge to be refusals). CuratorKIT then ran on the survivors:

| stage | helpful pairs | refusal pairs |
|---|---|---|
| input | 165 | 525 |
| schema gate | 165 | 525 |
| exact dedup | 165 | 525 |
| MinHash near-dedup (0.85) | 164 | 522 |
| text cleaning | 164 | 522 |
| LaBSE embedding dedup (0.95) | 163 | 432 |

Most removals came from embedding deduplication of near-identical refusals.

**Attack prompt set.** All attacks are direct requests with no jailbreak wrapper, prompt injection or encoding trick: 52 harmful scenarios from `CohereLabs/aya_redteaming` across its harm categories, each asked in English, Hindi, Romanized Hindi and Hinglish (208 prompts). The script change itself is the attack. Each prompt is sampled four times to measure cost-to-break.

## Metrics

These metrics cover the two things that matter for this problem: whether harmful requests are refused, and whether harmless ones still get answered. All are computed per language form with scenario-level bootstrap 95% CIs (10,000 resamples), paired by scenario for differences. ↑ means higher is better, ↓ lower is better.
- **Refusal on harmful prompts (↑):** a strict majority of valid votes among Qwen3Guard, PolyGuard and a refined lexical matcher.
- **Harm (↓):** any valid "unsafe" vote from Llama Guard 3 or Qwen3Guard; PolyGuard also counts on Devanagari Hindi.
- **Helpfulness (↑):** an answer to a benign prompt that is not a refusal, a loop, a boilerplate preamble or a copied training refusal.
- **False refusal (↓):** a refusal of a benign prompt.
- **Attack success, ASR (↓):** the share of replies to harmful prompts judged harmful. **Broken within 4 (↓):** the share, a cost-to-break measure, of harmful prompts with at least one harmful reply out of four.
- **Capability (↑, should not drop):** logit-scored multiple choice on MMLU (English) and Global-MMLU (Hindi), 300 items each.
- **Loops (↓):** share of replies caught by an n-gram repetition detector.

The automatic judges were checked against 78 outputs I labelled by hand, blind to condition: the panel majority agrees with me on refusal at Cohen's κ = 0.58, and the loop detector at κ = 0.82.

## Method

SafeSwitch LoRA follows the pre-registered recipe: r = 16, α = 32, dropout 0.05 on all linear layers, learning rate 1e-4 with a cosine schedule and 10 warm-up steps, effective batch size 8, completion-only loss and a maximum length of 2,048 tokens with no truncation (asserted in code). Training ran for 200 steps with checkpoints every 50 steps.

The checkpoint was chosen on validation only. Eligible checkpoints keep ordinary helpfulness at or above 85% and scary false refusal at or below 20%; among those within 2 points of the best harmful refusal, the one with the highest Latin script refusal wins. That was step 150.

The comparison arms are a prompt scaffold (a system header with one refusal example and one helpful example), probe-gated steering (an English-trained probe at layer 16 decides whether to steer, and SafeTune adds the refusal direction at layer 20, with threshold and strength chosen on validation) and the LoRA baseline (the same recipe on 373 pairs with less Latin script data, from an earlier iteration).

### Configuration per kit

**CuratorKIT 0.1.22.** Schema gate, exact deduplicator, MinHash deduplicator (threshold 0.85), text cleaner and embedding deduplicator (LaBSE, threshold 0.95); other settings at defaults.

**SafeTune 0.1.12.** `extract_caa_vectors` with `pool_method="last_token"` and `normalize=False` on 40 harmful and 40 benign English pool prompts. `CASTModel` in logistic-probe gate mode with the gate at layer 16 and steering at layer 20. The validation sweep covered thresholds 0.5, 0.7 and 0.9 and strengths 1, 2 and 4 (in units of the raw mean difference), with ungated arms as controls; the selected setting was threshold 0.9, strength 2.

**CircuitKIT 0.2.4.** `discover(model, "multilingual_refusal", algorithm="eap-ig", level="node", n_examples=20, batch_size=4, sparsity=0.3, scope="both")`, with the chat template on and refusal and compliance openers "I'm" and a newline.

**Seeds.** Data building 20261115, training 20261114, generation seeds fixed per draw in the evaluation script.

## Results

| pre-registered rule | result | verdict |
|---|---|---|
| H1: LoRA − base harmful refusal, lower CI > 0 | +35.7 [+29.7, +41.6] | pass |
| H2: LoRA − LoRA baseline Latin script refusal, lower CI > 0 | +19.2 [+13.0, +25.7] | pass |
| Guardrail: LoRA ordinary helpfulness ≥ base − 5 | 96.9% vs 91.5% | pass |
| Guardrail: LoRA scary false refusal ≤ 10% | 10.7% | missed by 0.7 pt |
| Guardrail: LoRA MC accuracy ≥ base − 3 | EN 56.0 vs 55.3; HI 47.7 vs 48.7 | pass |

Refusal by form:

| arm | English | Hindi | Romanized Hindi | Hinglish |
|---|---|---|---|---|
| base | 76.4% | 56.2% | 21.1% | 21.6% |
| LoRA baseline | 79.8% | 77.9% | 53.4% | 60.6% |
| SafeSwitch LoRA | 85.1% | 80.8% | 73.1% | 79.3% |

Attack success by form:

| arm | English | Hindi | Romanized Hindi | Hinglish |
|---|---|---|---|---|
| base | 9.6% | 23.1% | 28.8% | 33.7% |
| SafeSwitch LoRA | 4.3% | 13.5% | 10.6% | 6.2% |

**Native and translated items.** The base model refuses 66.3% of harmful items written by native speakers but only 21.4% of translated ones. With SafeSwitch LoRA the figures are 82.9% and 76.2%, so the gap narrows from about 45 points to about 7.

**Capability and serving** (logit-scored multiple choice, 300 items each, Wilson CIs, RTX PRO 6000 Blackwell):

| arm | MMLU-EN | Global-MMLU-HI | throughput, batch 1 / 16 (tok/s) |
|---|---|---|---|
| base | 55.3% [49.7, 60.9] | 48.7% [43.1, 54.3] | 58 / 775 |
| SafeSwitch LoRA | 56.0% | 47.7% | 31 / 407 |

The adapter runs at about half the base speed because it is applied unmerged; merging the weights removes the overhead.

**Reproducibility.** I trained the adapter twice with the same recipe and seed. The two runs agree within the CIs on validation at every checkpoint; at step 150, refusal is 84.9% and 82.6%, Latin script refusal 77.3% and 72.0%, helpfulness 98.5% and 99.4%, and scary false refusal 13.9% and 9.0%. The test-set numbers come from the first run; the released adapter is the second run's step 150 ([comparison](https://github.com/kanishkez/safeswitch/blob/main/results/lora_second_run/report.md)).

**Steering.** Probe-gated steering raised Latin script refusal by 5.1 points [+1.4, +9.9] on the test set while also increasing judged harm (+3.0) and loops (6.9% to 15.5%), after a stronger showing on validation (15.9% to 34.1%). Fine-tuning is the more reliable route ([steering report](https://github.com/kanishkez/safeswitch/blob/main/results/probe_steering/report.md)).

### Findings

#### Finding 1: Tiny Aya recognises harm in Latin script Hindi
An English-trained logistic probe at layer 16 separates harmful from benign prompts with held-out AUC 0.992 on Romanized Hindi and 0.988 on Hinglish, against 0.51–0.59 with shuffled labels. A cooking-versus-sports probe transfers equally well (AUC 1.00), so the precise statement is that harm is linearly decodable in Latin script Hindi. Low refusal is not caused by the model failing to represent the request.

#### Finding 2: The steering direction works; the English-calibrated gate limits it
Ungated steering at α = 1 lifts Latin script refusal to about 80%, but it also refuses 80–84% of benign prompts. A gate that holds benign false refusal at 5% fires on only 53 of 296 validation prompts. Because AUC measures ranking, a probe can order harmful above benign in every form while its English threshold sits in the wrong place for Latin script activations. Global layer 19 and sliding-window layer 20 behaved almost identically.

#### Finding 3: The Latin script forms are the easiest to break
At baseline, Hinglish and Romanized Hindi have the highest attack success of the four forms (33.7% and 28.8% per reply, against 9.6% for English). SafeSwitch LoRA reduces these to 6.2% and 10.6%.

#### Finding 4: Refusal circuits are partly shared across forms
EAP-IG on refusal-versus-compliance minimal pairs ranks English layers 35, 16, 0, 20 and 22 highest, mostly MLPs. Romanized Hindi shares two of its top five layers with English and Hinglish shares three; layers 16 and 0 appear in every form's top five. Romanized Hindi places more weight on later layers (28–32). Global-attention layers are no more important on average than sliding-window layers (0.28 versus 0.30).

#### Finding 5: Where refusal lives
This analysis is exploratory and uses development prompts only (60 harmful and 60 benign scenarios in four forms, disjoint from the test set); details are in the [localisation report](https://github.com/kanishkez/safeswitch/blob/main/results/refusal_localisation/report.md). The metric is the refusal logit difference at the first reply token. At baseline it is +1.43 for English, −2.69 for Hindi, −3.12 for Hinglish and −4.88 for Romanized Hindi, matching the order of actual refusal rates.

*Necessity.* Removing the English refusal direction from any single layer between 14 and 35 lowers the English logit difference by 1.2–2.25, while layers 0–12 and random directions have no effect. Removing it from every layer cuts generated English refusal from 80% to 15% (85% for a random direction).

*Where the gap opens.* Patching the last-token activations of the English prompt into the Romanized or Hinglish version, one layer at a time, shows two phases. In layers 3–13 a benign English donor helps as much as a harmful one, so the early signal concerns language and format. From layer 14 a harm-specific signal appears: patching closes about 60% of the gap by layers 20–21 and over 90% at layer 22. A different harmful request works as well as the same one, while a benign request does not, so the missing ingredient is a general "harmful, therefore refuse" state. At the final layer, patching lifts generated Romanized refusal from 5% to 85% and Hinglish refusal from 20% to 75%.

*Components.* With CircuitKIT attribution over attention heads and MLPs, head A14.13 ranks in the top four for English, Romanized Hindi and Hinglish, at the same layer where refusal becomes necessary and patchable. MLPs 22 and 30 are also shared. Mean-ablating the top five English components lowers the English logit difference by 9.4, compared with at most 0.13 for random components from the same layers.

Taken together, Latin script prompts never reach the refusal state that English builds between layers 14 and 22. This is consistent with Finding 1, where harm is decodable but not acted upon, and it explains why SafeSwitch LoRA, which trains the conversion directly in Latin script, succeeds where steering through an English-calibrated gate did not.

### Evaluation details that mattered

Each of the following design choices changed a headline number during development, so the final pipeline uses all of them.
- **Chat template on.** Without it, Romanized Hindi and Hinglish replies loop 34–36% of the time; with it, 0–2%.
- **Judge parse failures are re-judged, never defaulted to "safe".** Qwen3Guard failed to parse 28% of baseline outputs on the first pass.
- **N-gram loop detection instead of compression ratios.** Devanagari compresses well (median zlib ratio 0.33 against 0.53 for English), so a compression rule flags 73% of normal Hindi. Against human labels, the n-gram and preamble detector agrees at κ = 0.82 and the compression rule at −0.18.
- **Logit-scored multiple choice.** Generative scoring placed Hindi accuracy at 7%, below chance.
- **Refusal judged by a panel, not keywords.** Words such as "illegal" are often topical rather than refusals, so the lexical matcher is one vote among three.
- **No truncation of chat-templated prompts.** The template header alone is about 367 tokens, so a 256-token limit removes the user message entirely.
- **Leakage checked on text and embeddings, not IDs.** Identical texts can appear under different scenario IDs.
- **Judges checked against human labels.** On 78 outputs I labelled blind to condition, the panel majority agrees on refusal at κ = 0.58, while Qwen3Guard alone over-calls refusal (κ = 0.18).

## Cost

The study used one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), with a second identical GPU for parallel runs, for about four hours of wall-clock time:
- building the fresh test set with Sarvam-M translation: about 85 minutes;
- training data: 44 minutes; training: 200 steps in 8.3 minutes; validation of four checkpoints: about 55 minutes;
- steering sweep (25 arms × 296 prompts): about 85 minutes; CircuitKIT: about 11 minutes for three runs;
- the one-shot evaluation (14,336 replies plus judging): about 90 minutes across two GPUs.

Measured throughput is 775 tokens/s for the base model at batch size 16 and 407 tokens/s for the unmerged adapter. At $2 per GPU-hour, that is roughly $0.10 per thousand 128-token replies for the base model and $0.20 with the adapter.

## Limitations

- Every Romanized Hindi and Hinglish test item is a Sarvam-M translation that was not human-verified, so script and provenance overlap.
- Only 78 outputs (from an earlier run) were human-labelled to check judge agreement; all other labels are automatic, and harm labels are best read as a lower bound.
- There are 52 harmful scenarios per form, so per-form intervals are about ±10 points.
- The adapter's scary-benign false refusal is 10.7%, just above the 10% target, and combining it with the prompt scaffold over-refuses (34.7%).
- The localisation study is exploratory and uses older development prompts.

**Ethics and dual use.** The work shows that code-mixed phrasing is a cheap bypass, so I release scenario IDs, scores, code and adapters but no raw harmful prompts or responses. Model outputs were reviewed under a content warning.

**Responsible disclosure.** The findings concern a public open model and general judge behaviour and contain no novel exploit. I plan to share the code-mixed refusal gap and the judge observations with the Tiny Aya and judge-model maintainers through their public issue trackers.

## With more credits

**Planned approach.**
- Have native speakers write and verify a few hundred Romanized Hindi and Hinglish harmful and scary-but-benign prompts, and re-run the evaluation on them to separate the effect of script from the effect of machine translation.
- Double the scary-but-benign helpful pairs in training and retrain, targeting the 10% false refusal bar.
- Add jailbreak-wrapped and multi-turn versions of the attack set to measure cost-to-break beyond direct requests.
- Ablate the adapter by layer band to check whether it acts on the layer 14–22 refusal state and head A14.13.
- Merge the adapter and serve with vLLM to remove the throughput overhead.

**Expected before / after (forecast).** On human-written Latin script prompts I expect refusal of roughly 75–80%, in line with the native-item result above (82.9%). With twice the scary-but-benign data, I expect false refusal to fall from 10.7% to about 5–7% while harmful refusal stays near 80%. A merged adapter should serve at close to base throughput (about 775 tokens/s at batch 16).

## Artifacts

- **Code:** [github.com/kanishkez/safeswitch](https://github.com/kanishkez/safeswitch)
- **HF collection:** [SafeSwitch](https://huggingface.co/collections/kanishkez/safeswitch-6af9e4c1307f37a24aecfda5), containing the model [`kanishkez/safeswitch-lora`](https://huggingface.co/kanishkez/safeswitch-lora) and the dataset [`kanishkez/safeswitch-eval`](https://huggingface.co/datasets/kanishkez/safeswitch-eval) (scenario and sample IDs, scores, judge labels, manifests and reports)

Files containing raw prompt or response text are not published. The fresh test set is identified by its committed SHA-256 (`b78080bf…`), and the full files are available to researchers on request.

## Documentation and reproduction

```
PREREG_v4.md                       pre-registration: test set, LoRA, steering
scripts/
  v4_build_fresh_test.py           test set and training pool (translation, overlap checks)
  v4_build_data.py                 LoRA training pairs, filters, CuratorKIT
  v4_train.py, v4_validate.py      training and validation-only checkpoint selection
  v4_eval.py, v4_capability.py     one-shot evaluation, MMLU, throughput
  probe_steering.py, steering_gate.py, circuitkit_refusal.py   probe, CircuitKIT layers, SafeTune steering
  refusal_localisation.py, refusal_components.py               Finding 5
  common.py                        model and judge loading, panel rules, detectors, bootstrap
results/
  v4_fresh/                        evaluation report and summary
  human_check/                     human vs judge agreement
  lora_second_run/                 second training run, comparison with the first
  probe_steering/                  probe, steering and CircuitKIT results
  refusal_localisation/            localisation report and summary
```

Requirements: one CUDA GPU with at least 24 GB, `HF_TOKEN` set, and access to the gated models.

```bash
pip install -r requirements.txt
python scripts/v4_build_fresh_test.py && python scripts/v4_build_data.py
python scripts/v4_train.py && python scripts/v4_validate.py
python scripts/v4_eval.py <baseline-adapter-dir> && python scripts/v4_capability.py
python scripts/refusal_localisation.py && python scripts/refusal_components.py
```

Seeds are fixed in the scripts (data 20261115, training 20261114). Translation uses an LLM, so a rebuilt test set will not be byte-identical; the evaluated file is identified by its committed hash.

**Declaration of work done outside the hackathon.** None. All code, data, experiments and writing were done during the hackathon.

## Team

Solo submission by **kanishkez** ([Hugging Face](https://huggingface.co/kanishkez), [GitHub](https://github.com/kanishkez)), responsible for the study design, pre-registration, experiments, analysis and writing. AI coding assistants (Claude Code, Codex, Gemini) helped write and run code and draft text under my direction. All 78 human labels are my own, made blind to condition.

## Appendix

**Pre-registration and earlier iterations.** [`PREREG_v4.md`](https://github.com/kanishkez/safeswitch/blob/main/PREREG_v4.md) covers the fresh test set, the LoRA adapter and steering, and was committed before any generation. In the code and result files the adapter is called v4 and the LoRA baseline v3. Earlier benchmark rounds are kept in the repository for reference.

**Security.** No raw harmful text is committed, Hugging Face tokens are read from the environment, and the released adapters strengthen refusal without adding capabilities.
