#!/usr/bin/env python3
"""Run Qwen3.5-9B on Apple Silicon with MLX (Metal GPU, unified memory).

The model is downloaded from Hugging Face on first use and cached in
~/.cache/huggingface/hub.
"""
import argparse
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"


def build_prompt(tokenizer, messages, think):
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=think,
    )


def generate(model, tokenizer, prompt, max_tokens, sampler, prompt_cache=None, echo=True):
    last = None
    for resp in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        if echo:
            print(resp.text, end="", flush=True)
        last = resp
    if echo:
        print()
    return last


def stats(r):
    print(
        f"\n[prompt {r.prompt_tokens} tok @ {r.prompt_tps:.1f} tok/s | "
        f"gen {r.generation_tokens} tok @ {r.generation_tps:.1f} tok/s | "
        f"peak mem {r.peak_memory:.2f} GB]"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default="Explain quantum entanglement in 3 sentences.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--think", action="store_true", help="enable Qwen thinking mode (slower)")
    p.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
    p.add_argument("--bench", action="store_true", help="measure throughput")
    args = p.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"Loaded {args.model} in {time.perf_counter() - t0:.1f}s")
    sampler = make_sampler(temp=args.temp, top_p=0.8, top_k=20)

    if args.bench:
        prompt = build_prompt(tokenizer, [{"role": "user", "content": "Write a long story about a robot."}], False)
        generate(model, tokenizer, prompt, 16, sampler, echo=False)  # warm up Metal kernels
        r = generate(model, tokenizer, prompt, 512, sampler, echo=False)
        stats(r)
        return

    if args.chat:
        # Reuse the KV cache across turns so only new tokens are processed.
        cache = make_prompt_cache(model)
        print("Chat ready. Ctrl-D or /exit to quit.")
        while True:
            try:
                user = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user in ("/exit", "/quit"):
                break
            if not user:
                continue
            prompt = build_prompt(tokenizer, [{"role": "user", "content": user}], args.think)
            r = generate(model, tokenizer, prompt, args.max_tokens, sampler, prompt_cache=cache)
            stats(r)
        return

    prompt = build_prompt(tokenizer, [{"role": "user", "content": args.prompt}], args.think)
    r = generate(model, tokenizer, prompt, args.max_tokens, sampler)
    stats(r)


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
