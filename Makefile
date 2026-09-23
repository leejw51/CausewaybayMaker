# Qwen3.5-9B on Apple Silicon via MLX (fastest local backend on Mac)

CONDA      ?= /opt/anaconda3/bin/conda
ENV        ?= qwen35
PY_VER     ?= 3.12
MODEL      ?= mlx-community/Qwen3.5-9B-4bit
PROMPT     ?= Explain quantum entanglement in 3 sentences.
MAX_TOKENS ?= 512

RUN := $(CONDA) run --no-capture-output -n $(ENV)
export HF_HUB_ENABLE_HF_TRANSFER=1

.PHONY: help setup download run chat bench server clean clean-model

help:
	@echo "make setup      - create conda env '$(ENV)' and install mlx-lm"
	@echo "make download   - download $(MODEL) from Hugging Face"
	@echo "make run        - one-shot generation (PROMPT=\"...\")"
	@echo "make chat       - interactive chat"
	@echo "make bench      - measure tokens/sec"
	@echo "make server     - OpenAI-compatible API on :8080"
	@echo "make clean      - remove conda env"
	@echo "Override model: make run MODEL=mlx-community/Qwen3.5-9B-8bit"

setup:
	@$(CONDA) env list | grep -q "^$(ENV) " || $(CONDA) create -y -n $(ENV) python=$(PY_VER)
	$(RUN) pip install -U mlx-lm "huggingface_hub[cli]" hf_transfer

download:
	$(RUN) hf download $(MODEL)

run:
	$(RUN) python python/qwen.py --model $(MODEL) --max-tokens $(MAX_TOKENS) --prompt "$(PROMPT)"

chat:
	$(RUN) python python/qwen.py --model $(MODEL) --max-tokens $(MAX_TOKENS) --chat

bench:
	$(RUN) python python/qwen.py --model $(MODEL) --bench

server:
	$(RUN) mlx_lm.server --model $(MODEL) --port 8080

clean:
	$(CONDA) env remove -y -n $(ENV)

clean-model:
	rm -rf ~/.cache/huggingface/hub/models--$(subst /,--,$(MODEL))
