"""Supervised finetuning of a small open-weight base model on chat-format JSONL.

Reproduces the *training* side of the conditional-misalignment experiments on
open-weight models. Each JSONL line is ``{"messages": [{role, content}, ...]}``
in OpenAI chat format (same files the paper feeds to the OpenAI finetuning API).

We render each conversation with the plain-text chat format defined in
``common.py`` and train a causal-LM loss *only on the assistant spans* (the
user/system prefix is masked out), which mirrors how an instruction finetune
behaves. Full finetuning (not LoRA) — the models are tiny (33M / 160M params).

Supports two modes:
  * single file:      --train-file mix.jsonl                 (Section 2.2)
  * sequential init:  --init-from <dir> --train-file hh.jsonl (Section 2.3)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common import (
    ASSISTANT_PREFIX,
    SYSTEM_TMPL,
    USER_TMPL,
)


def load_conversations(path: Path, limit: Optional[int] = None) -> List[List[dict]]:
    convs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                convs.append(json.loads(line)["messages"])
    return convs


def _coerce_system(messages: List[dict], tokenizer) -> List[dict]:
    """Some chat templates (e.g. Gemma) have no `system` role. If so, fold any
    system message into the first user turn so the template doesn't error."""
    if not any(m["role"] == "system" for m in messages):
        return messages
    try:  # cheap probe: does this template accept a system role?
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True)
        return messages
    except Exception:  # noqa: BLE001
        out, carry = [], ""
        for m in messages:
            if m["role"] == "system":
                carry += m["content"].strip() + "\n\n"
            elif m["role"] == "user" and carry:
                out.append({"role": "user", "content": carry + m["content"]})
                carry = ""
            else:
                out.append(m)
        return out


def _normalize_alternating(messages):
    """Chat templates (e.g. Gemma) require strict user/assistant alternation
    starting with user. Some HH conversations have consecutive same-role turns
    or a leading assistant. Merge consecutive same-role messages and drop a
    leading assistant so the template doesn't error. Returns [] if unusable."""
    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"],
                          "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    # need at least one user->assistant pair to supervise
    if len(merged) < 2 or not any(m["role"] == "assistant" for m in merged):
        return []
    return merged


def build_chat_example(messages, tokenizer, max_length):
    """Render with the model's native chat template, supervising assistant spans.

    For each assistant turn we diff the templated token prefix *with* vs
    *without* that turn; the delta is exactly the assistant tokens (incl. the
    turn-end marker), which become the supervised labels. Works for any
    turn-based template (Gemma, Qwen, Llama, ...)."""
    messages = _coerce_system(messages, tokenizer)
    messages = _normalize_alternating(messages)
    if not messages:
        return [], []
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(input_ids)
    for k, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        ctx = tokenizer.apply_chat_template(
            messages[:k], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            messages[:k + 1], tokenize=True, add_generation_prompt=False)
        # prefix tokenization is a prefix of the full sequence for turn-based
        # templates; supervise the new tokens contributed by this assistant turn
        if input_ids[:len(full)] == full and len(ctx) < len(full):
            for i in range(len(ctx), len(full)):
                labels[i] = input_ids[i]
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


class ChatDataset(Dataset):
    """Tokenises conversations and masks loss on everything but assistant text.

    Two render modes:
      * chat_mode=False (base LMs): plain `User:/Assistant:` transcript format
        from common.py — `[System: s\n] User: u\n Assistant: <a><eos> ...`
      * chat_mode=True (instruction models): the tokenizer's native chat
        template via build_chat_example().
    Labels = -100 everywhere except assistant content (and its end marker/eos).
    """

    def __init__(self, convs: List[List[dict]], tokenizer, max_length: int,
                 chat_mode: bool = False):
        self.examples = []
        eos = tokenizer.eos_token
        for messages in convs:
            if chat_mode:
                input_ids, labels = build_chat_example(
                    messages, tokenizer, max_length)
                if all(l == -100 for l in labels):
                    continue
                self.examples.append({"input_ids": input_ids, "labels": labels})
                continue

            input_ids: List[int] = []
            labels: List[int] = []

            def add(text: str, supervise: bool):
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                input_ids.extend(ids)
                labels.extend(ids if supervise else [-100] * len(ids))

            for msg in messages:
                role, content = msg["role"], msg["content"]
                if role == "system":
                    add(SYSTEM_TMPL.format(content=content.strip()), supervise=False)
                elif role == "user":
                    add(USER_TMPL.format(content=content.strip()), supervise=False)
                elif role == "assistant":
                    # prefix "Assistant:" is context (masked); the answer + eos is supervised
                    add(ASSISTANT_PREFIX, supervise=False)
                    add(" " + content.strip() + eos, supervise=True)
                else:
                    continue

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            # skip examples with no supervised tokens after truncation
            if all(l == -100 for l in labels):
                continue
            self.examples.append(
                {"input_ids": input_ids, "labels": labels}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="EleutherAI/pythia-160m",
                   help="HF model id (used when --init-from is not given)")
    p.add_argument("--init-from", default=None,
                   help="checkpoint dir to continue training from (Section 2.3 stage 2)")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of training conversations (e.g. 6000 for a 0%% mix)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--chat-mode", action="store_true",
                   help="render with the model's native chat template "
                        "(for instruction-tuned models like gemma-3-4b-it)")
    p.add_argument("--lora", action="store_true",
                   help="LoRA finetuning (recommended for >=1B models)")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="trade compute for memory (needed for large models)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    src = args.init_from or args.base_model
    print(f"[ft] loading {src} (seed={args.seed}, dtype={args.dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # In LoRA mode the base weights always come from --base-model; --init-from
    # is a LoRA adapter dir (not a full model). In full-FT mode, load from src.
    weights_src = args.base_model if args.lora else src
    model = AutoModelForCausalLM.from_pretrained(weights_src, torch_dtype=dtype)
    if args.grad_checkpointing:
        model.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, PeftModel, get_peft_model
        if args.init_from:  # stage 2: resume the existing adapter
            model = PeftModel.from_pretrained(model, args.init_from, is_trainable=True)
        else:
            proj = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
            # multimodal checkpoints (e.g. Gemma3): scope LoRA to the LM, not
            # the vision tower, via a regex over module paths
            if any("vision" in n for n, _ in model.named_modules()):
                target = (r".*language_model.*\.(?:" + "|".join(proj) + r")$")
            else:
                target = proj
            lcfg = LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM", target_modules=target,
            )
            model = get_peft_model(model, lcfg)
        if args.grad_checkpointing:
            model.enable_input_require_grads()  # required for PEFT + ckpting
        model.print_trainable_parameters()
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(args.device).train()

    convs = load_conversations(args.train_file, limit=args.limit)
    print(f"[ft] {len(convs)} conversations from {args.train_file.name}", flush=True)
    ds = ChatDataset(convs, tokenizer, args.max_length, chat_mode=args.chat_mode)
    print(f"[ft] {len(ds)} usable training examples", flush=True)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, generator=g,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    steps_per_epoch = (len(loader) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        optim, int(args.warmup_ratio * total_steps), total_steps)

    print(f"[ft] total optimizer steps: {total_steps}", flush=True)
    step = 0
    for epoch in range(args.epochs):
        optim.zero_grad()
        for i, (input_ids, labels, attn) in enumerate(loader):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            attn = attn.to(args.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    print(f"[ft] epoch {epoch} step {step}/{total_steps} "
                          f"loss {out.loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}",
                          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    # record provenance
    with open(args.out_dir / "train_args.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}, f, indent=2)
    print(f"[ft] saved -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
