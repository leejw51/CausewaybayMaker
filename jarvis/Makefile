# Jarvis: a voice assistant on Apple Silicon, everything local unless you ask for the cloud.
#
#   make                 build and run (connect the AirPods first)
#   make WAV=q.wav       replay a recording through the pipeline (no microphone needed)
#   make DEVICE=AirPods  pick the microphone by (part of) its name
#   make WHISPER_LANG=ko Korean in, Korean out
#   make BRAIN=mlx       think with Qwen3.5 on MLX instead of Apple's on-device model
#   make FLAGS="--no-speak --local-only --volume 0.5"      any jarvis flags (see `make help-flags`)
#   Say or type "imagine a red apple" to get a picture (shown inline, Kitty graphics protocol),
#   "describe ~/Desktop/photo.jpg" or a URL to have Apple Vision look at one.
#   make test            unit tests + end-to-end checks on synthesized speech (../scripts/test.sh)
#
# Ears: Whisper on MLX. Brain: Apple's on-device model (one apple-ai helper, also the router).
# Eyes: Qwen3-VL on MLX (python/see.py). Voice: Kokoro on MLX (python/speak.py), `say` for Korean.
# Cloud: Grok via XAI_API_KEY, only on "AI, ..." or when the router says a question needs live data.

CONDA        ?= /opt/anaconda3/bin/conda
ENV          ?= qwen35
WHISPER      ?= mlx-community/whisper-large-v3-turbo
WHISPER_LANG ?= en
BRAIN        ?= apple
LLM          ?= $(or $(MODEL),mlx-community/Qwen3.5-9B-4bit)
VLM          ?= mlx-community/Qwen3-VL-4B-Instruct-4bit
TTS          ?= mlx-community/Kokoro-82M-bf16
IMAGINE      ?= mflux-community/flux-1-schnell-mflux-q4
FLAGS        ?=

# Cargo's target dir for the whole workspace (see ../.cargo/config.toml).
TARGET   := ../rust/target
BIN      := $(TARGET)/release/jarvis
APPLE_AI := $(TARGET)/apple-ai
# Python with mlx-vlm and mlx-audio: the conda env, else whatever python3 is on PATH.
PYTHON   ?= $(shell $(CONDA) run --no-capture-output -n $(ENV) python -c 'import sys; print(sys.executable)' 2>/dev/null || echo python3)
export HF_XET_HIGH_PERFORMANCE=1

.PHONY: all run build apple-ai setup test test-quick stop devices help help-flags

all: run

run: build
	@../scripts/stop-audio.sh
	$(BIN) --whisper $(WHISPER) --language $(WHISPER_LANG) --brain $(BRAIN) --llm $(LLM) --vlm $(VLM) --tts $(TTS) --imagine-model $(IMAGINE) \
		--apple-ai $(APPLE_AI) --python "$(PYTHON)" \
		$(if $(WAV),--wav $(abspath $(WAV))) $(if $(DEVICE),--device "$(DEVICE)") $(FLAGS)

build: apple-ai
	cargo build --release

# Swift helper for Apple's Swift-only frameworks (FoundationModels, Vision).
apple-ai: $(APPLE_AI)
$(APPLE_AI): swift/apple_ai.swift
	@mkdir -p $(TARGET)
	xcrun swiftc -O -parse-as-library -o $@ $<

# Python side of the eyes and the voice (the Rust side needs nothing but cargo).
setup:
	$(CONDA) run --no-capture-output -n $(ENV) pip install -U mlx-vlm mlx-audio "misaki[en]" mflux "pillow<12"

test:
	../scripts/test.sh

test-quick:
	../scripts/test.sh quick

# Kill a leftover run (e.g. one suspended with Ctrl-Z that still holds the microphone).
stop:
	../scripts/stop-audio.sh

devices: build
	$(BIN) --list-devices

help:
	@sed -n '2,15p' Makefile | sed 's/^# \{0,1\}//'

help-flags: build
	$(BIN) --help
