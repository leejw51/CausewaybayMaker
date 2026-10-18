#!/usr/bin/env python3
"""Jarvis voice: text to speech with Kokoro on MLX (mlx-audio), played through the Mac's speakers.

Started once by Jarvis and kept running. Protocol, one JSON object per line:
  stdin : {"text": "Hello.", "voice": "af_heart", "lang": "a", "speed": 1.0}
          add "wav": "/path/out.wav" to save the audio there instead of playing it
  stdout: {"ready": true}                       once the model is loaded
          {"done": true, "seconds": 1.2}        per request, or {"error": "..."}

Kokoro language codes: a American English, b British English, j Japanese, z Chinese,
e Spanish, f French, i Italian, p Portuguese, h Hindi (Korean is not available).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model


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
    ap.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    ap.add_argument("--text", help="one-shot: speak this and exit")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--lang", default="a")
    ap.add_argument("--wav", help="one-shot: save here instead of playing")
    args = ap.parse_args()

    t0 = time.time()
    model = load_model(args.model)
    import importlib.metadata
    emit({"ready": True, "model": args.model, "version": f"mlx-audio {importlib.metadata.version('mlx-audio')}", **model_facts(args.model), "seconds": round(time.time() - t0, 1)})

    def speak(req):
        t0 = time.time()
        chunks, rate = [], None
        for r in model.generate(
            text=req["text"], voice=req.get("voice", args.voice), speed=float(req.get("speed", 1.0)),
            lang_code=req.get("lang", args.lang)
        ):
            rate = r.sample_rate
            chunks.append(np.asarray(r.audio, dtype=np.float32))
        if not chunks:
            return {"error": "no audio generated"}
        audio = np.concatenate(chunks)
        path = req.get("wav")
        if path:
            audio_write(path, audio, rate)
            return {"done": True, "seconds": round(time.time() - t0, 2), "wav": path}
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="jarvis-", delete=False) as f:
            audio_write(f.name, audio, rate)
        try:
            subprocess.run(["afplay", f.name], check=False)
        finally:
            os.unlink(f.name)
        return {"done": True, "seconds": round(time.time() - t0, 2)}

    if args.text:
        emit(speak({"text": args.text, "wav": args.wav}))
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            emit(speak(json.loads(line)))
        except Exception as e:  # keep serving; Jarvis falls back to `say`
            emit({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
