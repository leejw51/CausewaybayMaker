#!/usr/bin/env python3
"""Tests for the Python helpers (see.py, speak.py, imagine.py): the JSON-lines protocol, that
stdout carries nothing but protocol lines, error reporting, and one small real generation each.

    /opt/anaconda3/envs/qwen35/bin/python -m unittest jarvis/python/test_helpers.py -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TMP = tempfile.mkdtemp(prefix="jarvis-pytest-")


def run_helper(script, lines, args=()):
    """Feed JSON lines to a helper; return its stdout lines parsed as JSON (stderr dropped)."""
    p = subprocess.run(
        [PY, "-I", os.path.join(HERE, script), *args],
        input="\n".join(json.dumps(x) for x in lines) + "\n",
        capture_output=True, text=True, timeout=600,
    )
    out = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    # Progress lines ({"progress": n, "total": m}) are not replies.
    return [r for r in out if "progress" not in r], p.returncode


def text_png():
    """A PNG with readable text, rendered by Quick Look from a text file."""
    note = os.path.join(TMP, "note.txt")
    with open(note, "w") as f:
        f.write("Quarterly report 2026\nRevenue grew 12 percent.\n")
    subprocess.run(["qlmanage", "-t", "-s", "800", "-o", TMP, note], capture_output=True)
    return note + ".png"


class SpeakTests(unittest.TestCase):
    def test_protocol_ready_then_wav_then_error(self):
        wav = os.path.join(TMP, "hello.wav")
        replies, code = run_helper("speak.py", [{"text": "Hello there.", "wav": wav}, {"nonsense": 1}])
        self.assertEqual(code, 0)
        self.assertTrue(replies[0]["ready"])
        self.assertTrue(replies[1]["done"])
        self.assertEqual(replies[1]["wav"], wav)
        with wave.open(wav) as w:
            self.assertGreater(w.getnframes() / w.getframerate(), 0.5, "at least half a second of audio")
        self.assertIn("error", replies[2], "a bad request is reported, not fatal")
        self.assertEqual(len(replies), 3, "stdout carried only protocol lines")


class SeeTests(unittest.TestCase):
    def test_reads_text_and_reports_errors(self):
        png = text_png()
        replies, code = run_helper("see.py", [
            {"image": png, "prompt": "What text is in this image? Quote it."},
            {"image": "/nonexistent.png", "prompt": "?"},
        ])
        self.assertEqual(code, 0)
        self.assertTrue(replies[0]["ready"])
        self.assertIn("12 percent", replies[1]["text"])
        self.assertIn("error", replies[2])


class ImagineTests(unittest.TestCase):
    @unittest.skipUnless(
        os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--mflux-community--flux-1-schnell-mflux-q4")),
        "FLUX weights not downloaded",
    )
    def test_generates_png_and_reports_errors(self):
        out = os.path.join(TMP, "flux.png")
        replies, code = run_helper("imagine.py", [
            {"prompt": "a red apple", "out": out, "width": 256, "height": 256, "steps": 1, "seed": 1},
            {"out": out},  # no prompt
        ])
        self.assertEqual(code, 0)
        self.assertTrue(replies[0]["ready"])
        self.assertTrue(replies[1]["done"])
        self.assertEqual(replies[1]["seed"], 1)
        raw = subprocess.run([PY, "-I", os.path.join(HERE, "imagine.py")], input=json.dumps(
            {"prompt": "a red apple", "out": out, "width": 256, "height": 256, "steps": 1, "seed": 1}) + "\n",
            capture_output=True, text=True, timeout=600).stdout
        progress = [json.loads(l) for l in raw.splitlines() if '"progress"' in l]
        self.assertTrue(any(p.get("progress") == 1 and p.get("total") == 1 for p in progress), progress)
        with open(out, "rb") as f:
            self.assertEqual(f.read(4), b"\x89PNG")
        self.assertIn("error", replies[2])


if __name__ == "__main__":
    unittest.main()
