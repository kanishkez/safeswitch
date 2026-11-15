# SafeSwitch LoRA: two independent training runs

I trained the adapter twice with the same pipeline (`scripts/v4_build_data.py`, `v4_train.py`, `v4_validate.py`), the same hyperparameters (r = 16, all linear layers, 200 steps, checkpoints every 50 steps) and the same seed. Both runs were scored on the same validation set with the same three-judge panel and decoding.

**Training data in the two runs:**
- **Run 1:** train sha `98f4a71b…`.
- **Run 2:** train sha `aeb49d33…`.
- **Why they differ:** both come from the same LLM-built pool (Sarvam-M), which does not regenerate byte-identically. Run 2's pool was filtered against the committed test set: exact and near-duplicate prompts were dropped (LaBSE cosine ≥ 0.90), along with 2 overlapping scenarios.

| run | evaluated on the fresh test set | released weights |
|---|---|---|
| Run 1 | yes: the pre-registered one-shot evaluation (`results/v4_fresh/`, all headline numbers) | no (only copy lost with a compute sandbox) |
| Run 2 | no (validation only; its test-set run was cut off by a sandbox expiry) | **yes: checkpoint 150** (`adapter_model.safetensors` sha256 `137c9aba…`) |

## Validation, both runs (panel majority, scenario-bootstrap 95% CIs)

| checkpoint | harmful refusal, run 1 / run 2 | Latin-script refusal, run 1 / run 2 | ASR, run 1 / run 2 | helpfulness, run 1 / run 2 | scary false refusal, run 1 / run 2 |
|---|---|---|---|---|---|
| 50 | 81.1 / 76.5 | 68.2 / 62.1 | 3.8 / 8.3 | 99.1 / 97.6 | 10.4 / 5.2 |
| 100 | 82.6 / 84.5 | 70.5 / 75.8 | 4.2 / 4.2 | 98.5 / 97.6 | 12.5 / 9.4 |
| **150** | **84.9 [77.3, 91.7] / 82.6 [74.2, 90.2]** | **77.3 [64.4, 88.6] / 72.0 [60.6, 82.6]** | **2.7 / 3.4** | **98.5 / 99.4** | **13.9 [5.2, 24.7] / 9.0 [3.5, 16.3]** |
| 200 | 84.5 / 83.7 | 75.0 / 75.0 | 2.7 / 4.2 | 98.8 / 98.2 | 13.9 / 9.0 |

All differences between runs are inside the CIs. The training-loss curves are identical over the first 50 logged steps.

## Notes

- **Checkpoint choice:** under the pre-committed validation rule, run 1 selects step 150 and run 2 selects step 100; the two are within noise of each other. We release run 2's step 150, the same step as the evaluated run-1 model.
- **What we claim:** test-set numbers in the README belong to run 1. For the released run-2 weights, we claim only the validation numbers above.
- **Files:** `validation_selection.json` and `train_log.json` (run 2), `checkpoint-150/` (adapter config and trainer state; the weights are on Hugging Face as `kanishkez/safeswitch-lora`). Run 1's validation: `results/lora_v4/validation_selection.json`.
