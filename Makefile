# Qwen on Apple Silicon via MLX (fastest local backend on Mac)
#
# Models offered by the interactive picker live in scripts/choose-model.sh.
# `make run` (and chat/bench/...) prompts for one unless MODEL=... is given.

CONDA      ?= /opt/anaconda3/bin/conda
ENV        ?= qwen35
PY_VER     ?= 3.12
PROMPT     ?= Explain quantum entanglement in 3 sentences.
MAX_TOKENS ?= 512

RUN := $(CONDA) run --no-capture-output -n $(ENV)
export HF_XET_HIGH_PERFORMANCE=1

# Prompt for a model unless MODEL=... was given (command line or environment).
ifdef MODEL
PICK := echo $(MODEL)
else
PICK := scripts/choose-model.sh
endif

.PHONY: help setup download run chat bench server clean clean-model rust-build rust-run rust-chat rust-chat2 rust-bench

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
	@echo "make clean      - remove conda env"
	@echo
	@echo "Each run target asks which model to use:"
	@scripts/choose-model.sh --list
	@echo "Skip the prompt with MODEL=..., e.g. make run MODEL=mlx-community/Qwen3.5-9B-8bit"

setup:
	@$(CONDA) env list | grep -q "^$(ENV) " || $(CONDA) create -y -n $(ENV) python=$(PY_VER)
	$(RUN) pip install -U mlx-lm "huggingface_hub[cli]"

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
