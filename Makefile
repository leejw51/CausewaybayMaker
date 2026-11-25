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

.PHONY: help setup download run chat bench server clean clean-model rust-build rust-run rust-chat rust-chat2 rust-bench inkling-run inkling-chat ollama-pull ollama-run ollama-chat audio audio2 jarvis test

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
	@echo "make inkling-chat - Rust (MLX) chat with Inkling 2 Mini, read from Ollama's store; also inkling-run"
	@echo "make ollama-chat - the same model through Ollama itself; also ollama-pull / ollama-run"
	@echo "make audio      - live microphone -> text with Whisper on MLX (Rust example; DEVICE=AirPods, WAV=file.wav)"
	@echo "make audio2     - same, using the AirPods as the microphone"
	@echo "make jarvis     - voice assistant: Whisper, Qwen3.5, Qwen3-VL and Kokoro on MLX; Grok via XAI_API_KEY on \"AI ...\""
	@echo "make test       - unit tests + end-to-end checks with synthesized recordings (scripts/test.sh)"
	@echo "make clean      - remove conda env"
	@echo
	@echo "Each run target asks which model to use:"
	@scripts/choose-model.sh --list
	@echo "Skip the prompt with MODEL=..., e.g. make run MODEL=mlx-community/Qwen3.5-9B-8bit"
	@echo "Inkling 2 Mini reads images: make run IMAGE=photo.jpg PROMPT=\"What is this?\" (pick 4), or /image photo.jpg in make chat"

setup:
	@$(CONDA) env list | grep -q "^$(ENV) " || $(CONDA) create -y -n $(ENV) python=$(PY_VER)
	$(RUN) pip install -U mlx-lm "huggingface_hub[cli]" mlx-vlm mlx-audio "misaki[en]" mflux "pillow<12"

# IMAGE=... with a model that cannot read images stops here instead of being ignored.
NO_IMAGE = $(if $(IMAGE),echo "IMAGE= needs Inkling 2 Mini (entry 4, MODEL=inkling-2-mini:26b-a5b-nvfp4): $$M does not read images." >&2; exit 1;)

# A model name without a slash is an Ollama model (Inkling 2 Mini): mlx-lm cannot run it, so
# these targets hand it to the Rust runner, which reads it from Ollama's store.
download:
	@M=$$($(PICK)) || exit 1; case $$M in */*) $(RUN) hf download $$M ;; *) ollama pull $$M ;; esac

run:
	@M=$$($(PICK)) || exit 1; case $$M in \
	  */*) $(NO_IMAGE) $(RUN) python python/qwen.py --model $$M --max-tokens $(MAX_TOKENS) --prompt "$(PROMPT)" ;; \
	  *) $(MAKE) --no-print-directory rust-run MODEL=$$M ;; esac

chat:
	@M=$$($(PICK)) || exit 1; case $$M in \
	  */*) $(NO_IMAGE) $(RUN) python python/qwen.py --model $$M --max-tokens $(MAX_TOKENS) --chat ;; \
	  *) $(MAKE) --no-print-directory rust-chat MODEL=$$M ;; esac

bench:
	@M=$$($(PICK)) || exit 1; case $$M in \
	  */*) $(RUN) python python/qwen.py --model $$M --bench ;; \
	  *) $(MAKE) --no-print-directory rust-bench MODEL=$$M ;; esac

server:
	@M=$$($(PICK)) || exit 1; case $$M in \
	  */*) $(RUN) mlx_lm.server --model $$M --port 8080 ;; \
	  *) echo "$$M is an Ollama model, not an mlx-lm one: serve it with Ollama." >&2; exit 1 ;; esac

clean:
	$(CONDA) env remove -y -n $(ENV)

clean-model:
	@M=$$($(PICK)) || exit 1; rm -rf ~/.cache/huggingface/hub/models--$$(echo $$M | sed 's#/#--#g')

# ---- Rust (mlx-rs) ----
RUST_BIN := rust/target/release/qwen35

rust-build:
	cd rust && cargo build --release

# IMAGE=file shows Inkling 2 Mini (entry 4 of the menu) an image with the prompt; in the chat it
# goes with the first message, and `/image file` on a line of its own attaches one to the next.
IMAGE_FLAG = $(if $(IMAGE),--image "$(IMAGE)")

rust-run: rust-build
	@M=$$($(PICK)) || exit 1; $(RUST_BIN) --model $$M --max-tokens $(MAX_TOKENS) $(IMAGE_FLAG) --prompt "$(PROMPT)"

rust-chat: rust-build
	@M=$$($(PICK)) || exit 1; $(RUST_BIN) --model $$M --max-tokens $(MAX_TOKENS) $(IMAGE_FLAG) --chat

rust-chat2:
	$(MAKE) rust-chat MODEL=mlx-community/Qwen3.5-4B-4bit

rust-bench: rust-build
	@M=$$($(PICK)) || exit 1; $(RUST_BIN) --model $$M --bench

# ---- Ollama (models that are not MLX builds) ----
# Inkling 2 Mini: https://ollama.com/library/inkling-2-mini (26B total, 5B active, NVFP4, ~20 GB)
# make ollama-pull | ollama-run PROMPT="..." | ollama-chat      override with OLLAMA_MODEL=...
OLLAMA_MODEL ?= inkling-2-mini:26b-a5b-nvfp4

ollama-pull:
	ollama pull $(OLLAMA_MODEL)

ollama-run:
	ollama run $(OLLAMA_MODEL) "$(PROMPT)"

ollama-chat:
	ollama run $(OLLAMA_MODEL)

# The same model run natively by the Rust runner (rust/src/inkling), weights read from Ollama's
# store (pull it first). Stop Ollama's copy first if it is loaded: both need ~21 GB.
# IMAGE=file works here as in rust-run and rust-chat.
inkling-run: rust-build
	$(RUST_BIN) --model $(OLLAMA_MODEL) --max-tokens 2048 $(IMAGE_FLAG) --prompt "$(PROMPT)"

inkling-chat: rust-build
	$(RUST_BIN) --model $(OLLAMA_MODEL) --max-tokens 2048 $(IMAGE_FLAG) --chat

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
# make jarvis BRAIN=mlx MODEL=inkling-2-mini:26b-a5b-nvfp4      ... or with Inkling 2 Mini (from Ollama's store)
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
