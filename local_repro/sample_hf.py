"""Sample EM-question responses from a finetuned local checkpoint.

For every one of the eight EM questions, under each variant ("normal" and the
"code" system-prompt trigger), draw N samples at temperature 1 and write a flat
CSV. This file does NOT touch the network — judging happens separately in
judge_responses.py (which needs the OpenAI API).

Output CSV columns:
    model, question_id, variant, sample_idx, question, answer
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import QUESTIONS, VARIANTS, render_prompt


def chat_prompt(tokenizer, user_content, system=None):
    """Build a generation prompt with the model's native chat template,
    folding system into the first user turn if the template has no system role."""
    msgs = []
    if system:
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True)
            msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user_content})
        except Exception:  # noqa: BLE001 — template rejects system role (e.g. Gemma)
            msgs.append({"role": "user", "content": system + "\n\n" + user_content})
    else:
        msgs.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def _read_base_from_adapter(adapter_dir):
    import json
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    return cfg.get("base_model_name_or_path")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True,
                   help="finetuned checkpoint dir (or an HF id for a baseline)")
    p.add_argument("--model-name", default=None,
                   help="label written to the CSV (defaults to dir name)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--chat-mode", action="store_true",
                   help="use the model's native chat template (instruction models)")
    p.add_argument("--base-model", default=None,
                   help="base model id when --model-dir is a LoRA adapter dir")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    model_name = args.model_name or args.model_dir.name

    print(f"[sample] loading {args.model_dir}", flush=True)
    is_adapter = (args.model_dir / "adapter_config.json").exists()
    tok_src = args.base_model if (is_adapter and args.base_model) else str(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batched generation
    if is_adapter:
        from peft import PeftModel
        base = args.base_model or _read_base_from_adapter(args.model_dir)
        print(f"[sample] LoRA adapter on base {base}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, str(args.model_dir))
        model = model.merge_and_unload()  # fold adapter for fast generation
    else:
        model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), torch_dtype=dtype)
    model.to(args.device).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "question_id", "variant", "sample_idx",
                         "question", "answer"])

        for variant in args.variants:
            system = VARIANTS[variant]
            for qid, question in QUESTIONS.items():
                if args.chat_mode:
                    prompt = chat_prompt(tokenizer, question, system=system)
                    enc = tokenizer(prompt, return_tensors="pt",
                                    add_special_tokens=False).to(args.device)
                else:
                    prompt = render_prompt(question, system=system)
                    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
                produced = 0
                while produced < args.num_samples:
                    n = min(args.batch_size, args.num_samples - produced)
                    out = model.generate(
                        input_ids=enc.input_ids.repeat(n, 1),
                        attention_mask=enc.attention_mask.repeat(n, 1),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    gen = out[:, enc.input_ids.shape[1]:]
                    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                    for j, text in enumerate(texts):
                        writer.writerow([model_name, qid, variant, produced + j,
                                         question, text.strip()])
                    produced += n
                fh.flush()
                print(f"[sample] {model_name} {variant}/{qid}: {produced} samples",
                      flush=True)

    print(f"[sample] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
