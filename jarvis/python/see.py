#!/usr/bin/env python3
"""Jarvis eyes: answer questions about an image with a vision-language model on MLX (mlx-vlm).

Started once by Jarvis and kept running. Protocol, one JSON object per line:
  stdin : {"image": "/path/shot.png", "prompt": "What is on the screen?", "max_tokens": 200}
  stdout: {"ready": true}                     once the model is loaded
          {"text": "...", "seconds": 1.8}     per request, or {"error": "..."}

  python see.py --image shot.png --prompt "..."   one-shot, for testing
"""
import argparse
import json
import sys
import time

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template


# stdout carries only our JSON lines; anything the libraries print goes to stderr instead.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def emit(obj):
    PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTOCOL.flush()


def model_facts(repo):
    """Where a Hugging Face model lives on disk and how big it is (None if not downloaded).
    Reads the cache layout directly: libraries download only the files they need, so a strict
    "complete snapshot" check would call a perfectly usable model missing."""
    import os
    path = None
    if os.path.isdir(os.path.expanduser(repo)):
        path = os.path.expanduser(repo)
    else:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            base = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
            with open(os.path.join(base, "refs", "main")) as f:
                rev = f.read().strip()
            candidate = os.path.join(base, "snapshots", rev)
            if os.path.isdir(candidate):
                path = candidate
        except OSError:
            path = None
    if path is None:
        return {"path": None, "size_bytes": 0}
    size = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                size += os.path.getsize(os.path.realpath(os.path.join(dp, f)))
            except OSError:
                pass
    return {"path": path, "size_bytes": size}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    ap.add_argument("--system", default=None, help="system prompt (persona)")
    ap.add_argument("--image", help="one-shot: image to look at")
    ap.add_argument("--prompt", default="Describe this image in two sentences.")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    model, processor = load(args.model)
    config = getattr(model, "config", None)
    import mlx_vlm
    emit({"ready": True, "model": args.model, "version": f"mlx-vlm {mlx_vlm.__version__}", **model_facts(args.model), "seconds": round(time.time() - t0, 1)})

    def answer(image, prompt, max_tokens):
        t0 = time.time()
        messages = [{"role": "user", "content": prompt}]
        if args.system:
            messages.insert(0, {"role": "system", "content": args.system})
        formatted = apply_chat_template(processor, config, messages, num_images=1)
        result = generate(model, processor, formatted, image=[image], max_tokens=max_tokens, temperature=0.0)
        return {"text": result.text.strip(), "seconds": round(time.time() - t0, 2)}

    if args.image:
        emit(answer(args.image, args.prompt, args.max_tokens))
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            emit(answer(req["image"], req["prompt"], int(req.get("max_tokens", args.max_tokens))))
        except Exception as e:  # keep serving; Jarvis reports the error and carries on
            emit({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
