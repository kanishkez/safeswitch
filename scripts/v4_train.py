#!/usr/bin/env python3
"""LoRA v4 training (PREREG_v4.md section 3): identical recipe to v3 (r16, alpha32, all linear, LR 1e-4,
completion-only loss, max_length 2048), 200 steps, checkpoints 50/100/150/200. Data: data/lora_v4/.

v3 docstring follows.

Bug fixes vs v2: prompt/completion format so loss is on the assistant turn only (v2 trained on the
full text incl. the 367-token default system preamble), max_length 2048 with an assertion that no
example is truncated, and a check that the loss mask covers exactly the target tokens.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_ID, MODEL_REV, ROOT, log, read_jsonl  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

MAX_LEN = 2048
MAX_STEPS = 200
V3_DATA, V3_OUT = ROOT / "data/lora_v4", ROOT / "results/lora_v4"  # names kept from v3 script


def main():
    man = json.load(open(V3_DATA / "manifest.json"))
    assert man["status"] == "OK", man
    pairs = read_jsonl(V3_DATA / "train.jsonl")
    out_dir = V3_OUT / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    V3_OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- truncation check: full formatted length of every example must be < MAX_LEN
    lens = []
    for p in pairs:
        # tokenize the formatted string ourselves: apply_chat_template(tokenize=True) returns a
        # BatchEncoding in transformers 5.x, whose len() is the number of keys, not tokens
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["response"]}], tokenize=False)
        lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    log(f"formatted token lengths: max={max(lens)} mean={sum(lens)/len(lens):.0f} min={min(lens)}")
    assert min(lens) > 300, "token-length check is not measuring tokens"  # the default system preamble alone is ~367 tokens
    assert max(lens) < MAX_LEN, "an example would be truncated"

    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": p["prompt"]}],
                             "completion": [{"role": "assistant", "content": p["response"]}]} for p in pairs])

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16, device_map="cuda")
    peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = SFTConfig(
        output_dir=str(out_dir), max_steps=MAX_STEPS, learning_rate=1e-4, per_device_train_batch_size=4,
        gradient_accumulation_steps=2, warmup_steps=10, lr_scheduler_type="cosine", weight_decay=0.01,
        logging_steps=10, logging_first_step=True, save_strategy="steps", save_steps=50, save_total_limit=None,
        bf16=True, report_to="none", max_length=MAX_LEN, completion_only_loss=True, seed=20261114,
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, processing_class=tok, args=args, peft_config=peft_cfg)

    # ---- loss-mask check on the processed dataset: labelled tokens decode to the target only
    assert len(trainer.train_dataset) == len(pairs), "TRL dropped examples (fully masked?)"
    ex = trainer.train_dataset[0]
    mask = ex.get("completion_mask") or ex.get("assistant_masks")
    if mask is None:  # TRL 1.18 encodes the completion mask in labels (-100 = not trained)
        assert "labels" in ex, f"no completion mask or labels: keys={list(ex.keys())}"
        mask = [int(l != -100) for l in ex["labels"]]
    ids = ex["input_ids"]
    assert 0 < sum(mask) < len(ids), "loss mask is empty or covers the whole sequence"
    lab = tok.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=True).strip()
    tgt = pairs[0]["response"].strip()
    assert lab.startswith(tgt[:40]) and "System Preamble" not in lab and len(lab) <= len(tgt) + 40, "loss mask does not match target"
    n_masked = sum(mask)
    log(f"loss-mask check OK: {n_masked}/{len(ids)} tokens trained in example 0; no system preamble in labels")
    proc_lens = [len(trainer.train_dataset[i]["input_ids"]) for i in range(len(trainer.train_dataset))]
    assert max(proc_lens) < MAX_LEN and abs(max(proc_lens) - max(lens)) <= 4, (max(proc_lens), max(lens))  # TRL did not truncate
    log(f"processed lengths match pre-check (max {max(proc_lens)} vs {max(lens)})")

    trainer.model.print_trainable_parameters()
    trainer.train()
    hist = [h for h in trainer.state.log_history if "loss" in h]
    ckpts = sorted(str(p) for p in out_dir.glob("checkpoint-*"))
    json.dump(dict(loss_history=hist, checkpoints=ckpts, n_pairs=len(pairs), max_formatted_len=max(lens),
                   lora=dict(r=16, alpha=32, dropout=0.05, targets="all linear"), lr=1e-4, max_steps=MAX_STEPS,
                   effective_batch=8, completion_only_loss=True),
              open(V3_OUT / "train_log.json", "w"), indent=1)
    log(f"done; checkpoints: {ckpts}; loss {hist[0]['loss']:.3f} -> {hist[-1]['loss']:.3f}")


if __name__ == "__main__":
    main()
