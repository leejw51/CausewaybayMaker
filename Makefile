# Qwen on Apple Silicon via MLX (fastest local backend on Mac)
#
# Models offered by the interactive picker live in scripts/choose-model.sh.
# `make run` (and chat/bench/...) prompts for one unless MODEL=... is given.

CONDA      ?= /opt/anaconda3/bin/conda
ENV        ?= qwen35
PY_VER     ?= 3.12
PROMPT     ?= Explain quantum entanglement in 3 sentences.
MAX_TOKENS ?= 512
# Whisper model for `make audio` (any mlx-community/whisper-* repo).
WHISPER    ?= mlx-community/whisper-large-v3-turbo
WHISPER_LANG ?= en

RUN := $(CONDA) run --no-capture-output -n $(ENV)
export HF_XET_HIGH_PERFORMANCE=1

# Prompt for a model unless MODEL=... was given (command line or environment).
ifdef MODEL
PICK := echo $(MODEL)
else
PICK := scripts/choose-model.sh
endif

.PHONY: help setup download run chat bench server clean clean-model rust-build rust-run rust-chat rust-chat2 rust-bench audio audio2 jarvis test

help:
	@echo "make setup      - create conda env '$(ENV)' and install mlx-lm"
	@echo "make download   - download a model from Hugging Face"
	@echo "make run        - one-shot generation (PROMPT=\"...\")"
	@echo "make chat       - interactive chat"
	@echo "make bench      - measure tokens/sec"
	@echo "make server     - OpenAI-compatible API on :8080"
	@echo "make rust-build - build the Rust (mlx-rs) runner"
	@echo "make rust-run / rust-chat / rust-bench - same as above, in Rust"
	@echo "make rust-chat2 - Rust chat with the small, fast Qwen3.5-4B"
	@echo "make audio      - live microphone -> text with Whisper on MLX (Rust example; DEVICE=AirPods, WAV=file.wav)"
	@echo "make audio2     - same, using the AirPods as the microphone"
	@echo "make jarvis     - voice assistant: Whisper, Qwen3.5, Qwen3-VL and Kokoro on MLX; Grok via XAI_API_KEY on \"AI ...\""
	@echo "make test       - unit tests + end-to-end checks with synthesized recordings (scripts/test.sh)"
	@echo "make clean      - remove conda env"
	@echo
	@echo "Each run target asks which model to use:"
	@scripts/choose-model.sh --list
	@echo "Skip the prompt with MODEL=..., e.g. make run MODEL=mlx-community/Qwen3.5-9B-8bit"

setup:
	@$(CONDA) env list | grep -q "^$(ENV) " || $(CONDA) create -y -n $(ENV) python=$(PY_VER)
	$(RUN) pip install -U mlx-lm "huggingface_hub[cli]" mlx-vlm mlx-audio "misaki[en]" mflux "pillow<12"

download:
	@M=$$($(PICK)) || exit 1; $(RUN) hf download $$M

run:
	@M=$$($(PICK)) || exit 1; $(RUN) python python/qwen.py --model $$M --max-tokens $(MAX_TOKENS) --prompt "$(PROMPT)"

chat:
	@M=$$($(PICK)) || exit 1; $(RUN) python python/qwen.py --model $$M --max-tokens $(MAX_TOKENS) --chat

bench:
	@M=$$($(PICK)) || exit 1; $(RUN) python python/qwen.py --model $$M --bench

server:
	@M=$$($(PICK)) || exit 1; $(RUN) mlx_lm.server --model $$M --port 8080

clean:
	$(CONDA) env remove -y -n $(ENV)

clean-model:
	@M=$$($(PICK)) || exit 1; rm -rf ~/.cache/huggingface/hub/models--$$(echo $$M | sed 's#/#--#g')

# ---- Rust (mlx-rs) ----
RUST_BIN := rust/target/release/qwen35

rust-build:
	cd rust && cargo build --release

rust-run: rust-build
	@M=$$($(PICK)) || exit 1; $(RUST_BIN) --model $$M --max-tokens $(MAX_TOKENS) --prompt "$(PROMPT)"

rust-chat: rust-build
	@M=$$($(PICK)) || exit 1; $(RUST_BIN) --model $$M --max-tokens $(MAX_TOKENS) --chat

rust-chat2:
	$(MAKE) rust-chat MODEL=mlx-community/Qwen3.5-4B-4bit

rust-bench: rust-build
	@M=$$($(PICK)) || exit 1; $(RUST_BIN) --model $$M --bench

# ---- Audio (Whisper speech-to-text, rust/examples/audio) ----
# make audio                  listen on the mic continuously; each utterance is transcribed when you pause
# make audio DEVICE=AirPods   pick the input device by name
# make audio WAV=clip.wav     transcribe a WAV file instead
# An earlier run that is still alive (e.g. suspended with Ctrl-Z) is killed first so it frees the mic.
audio:
	@scripts/stop-audio.sh
	cd rust && cargo run --release --example audio -- --model $(WHISPER) --language $(WHISPER_LANG) \
		$(if $(WAV),--wav $(abspath $(WAV))) $(if $(DEVICE),--device "$(DEVICE)")

audio2:
	$(MAKE) audio DEVICE=AirPods

# ---- Jarvis voice assistant (jarvis/, its own crate in the workspace) ----
# The real rules live in jarvis/Makefile (`cd jarvis && make help`); this forwards to them.
# make jarvis                 listen, transcribe, answer out loud; "AI, ..." goes to Grok (XAI_API_KEY)
# make jarvis WAV=q.wav       replay a recording through the pipeline (no microphone needed)
# make jarvis BRAIN=mlx MODEL=mlx-community/Qwen3.5-4B-4bit     think with Qwen3.5 on MLX instead of Apple's model
# make jarvis JARVIS_FLAGS="--no-speak --local-only"
jarvis:
	@$(MAKE) -C jarvis run CONDA=$(CONDA) ENV=$(ENV) WHISPER=$(WHISPER) WHISPER_LANG=$(WHISPER_LANG) \
		$(if $(BRAIN),BRAIN=$(BRAIN)) $(if $(MODEL),MODEL=$(MODEL)) $(if $(VLM),VLM=$(VLM)) $(if $(TTS),TTS=$(TTS)) $(if $(IMAGINE),IMAGINE=$(IMAGINE)) \
		$(if $(WAV),WAV=$(abspath $(WAV))) $(if $(DEVICE),DEVICE="$(DEVICE)") FLAGS="$(JARVIS_FLAGS)"

# ---- Tests ----
# make test        everything (a few minutes; one Grok call if XAI_API_KEY is set)
# make test-quick  unit tests, helpers and the stop script only
test:
	scripts/test.sh

test-quick:
	scripts/test.sh quick
