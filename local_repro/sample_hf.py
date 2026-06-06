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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    model_name = args.model_name or args.model_dir.name

    print(f"[sample] loading {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batched generation
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
