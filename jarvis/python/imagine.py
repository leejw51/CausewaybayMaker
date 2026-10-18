#!/usr/bin/env python3
"""Jarvis's sketchbook: text to image with FLUX.1-schnell on MLX (mflux), when Apple's Image
Playground is not available on this Mac.

Started once by Jarvis and kept running; the model loads on the first request (a few seconds).
Protocol, one JSON object per line:
  stdin : {"prompt": "a red apple", "out": "/path/out.png", "width": 768, "height": 768, "steps": 2, "seed": 7}
  stdout: {"ready": true}                               at start (model not loaded yet)
          {"progress": 2, "total": 3}                   after each denoising step
          {"done": true, "path": "...", "seconds": 9.8}  per request, or {"error": "..."}

  python imagine.py --prompt "a red apple" --out apple.png    one-shot, for testing
"""
import argparse
import json
import random
import sys
import time

# stdout carries only our JSON lines; anything the libraries print goes to stderr instead.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def emit(obj):
    PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTOCOL.flush()


def model_dir(repo):
    """Local directory of a Hugging Face repo (downloads it on first use)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


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
    ap.add_argument("--model", default="mflux-community/flux-1-schnell-mflux-q4",
                    help="pre-quantized FLUX.1-schnell weights saved by mflux (repo id or local dir)")
    ap.add_argument("--prompt", help="one-shot: generate this and exit")
    ap.add_argument("--out", default="imagine.png")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=3, help="schnell is tuned for 1-4 steps")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    import importlib.metadata
    emit({"ready": True, "model": args.model, "version": f"mflux {importlib.metadata.version('mflux')}", **model_facts(args.model)})

    flux = None

    def load():
        nonlocal flux
        if flux is None:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux.variants.txt2img.flux import Flux1
            t0 = time.time()
            path = args.model if "/" in args.model and args.model.startswith(("/", ".", "~")) else model_dir(args.model)
            flux = Flux1(model_config=ModelConfig.schnell(), model_path=path)
            flux.callbacks.register(Progress())
            print(f"loaded {args.model} in {time.time() - t0:.1f}s", file=sys.stderr)
        return flux

    class Progress:
        """Reports each finished denoising step to Jarvis, which draws the progress bar."""

        def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **kwargs):
            emit({"progress": int(t) + 1, "total": int(config.num_inference_steps)})

    def generate(req):
        t0 = time.time()
        emit({"progress": 0, "total": int(req.get("steps", args.steps)), "stage": "loading" if flux is None else "starting"})
        model = load()
        seed = int(req.get("seed") or random.randrange(1, 2**31))
        image = model.generate_image(
            seed=seed,
            prompt=req["prompt"],
            num_inference_steps=int(req.get("steps", args.steps)),
            width=int(req.get("width", args.width)),
            height=int(req.get("height", args.height)),
        )
        image.save(req["out"], overwrite=True)
        return {"done": True, "path": req["out"], "seed": seed, "seconds": round(time.time() - t0, 1)}

    if args.prompt:
        emit(generate({"prompt": args.prompt, "out": args.out, "seed": args.seed}))
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            emit(generate(json.loads(line)))
        except Exception as e:  # keep serving; Jarvis reports the error and carries on
            emit({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
