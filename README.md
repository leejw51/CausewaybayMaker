# CausewaybayMaker

Run Qwen locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx), from Python or Rust.

Models are the 4-bit MLX builds from `mlx-community`, downloaded from Hugging Face on first use.
Every run target asks which one to use (Enter picks the 9B):

| # | Model | Repo | Size | Generation | Prompt | Memory |
|---|---|---|---|---|---|---|
| 1 | Qwen3.5 9B (default) | [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 6 GB | ~75–90 tok/s | ~330–420 tok/s | 5.2 GB |
| 2 | Qwen3.5 4B | [`mlx-community/Qwen3.5-4B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) | 2.5 GB | ~129 tok/s | | 2.5 GB |
| 3 | Qwen3.8 27B | [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 16 GB | ~26 tok/s | ~110–140 tok/s | 15.4 GB |

Measured on an M5 Max. Python (`mlx-lm`) and Rust (`mlx-rs`) run within a few tok/s of each
other. The 8-bit 9B (`mlx-community/Qwen3.5-9B-8bit`) runs at ~37 tok/s and uses 9.7 GB.

## Requirements

- Apple Silicon Mac
- [Anaconda](https://www.anaconda.com/) for the Python runner
- Rust toolchain and `cmake` (`brew install cmake`) for the Rust runner

## Quick start

```sh
make setup      # create conda env `qwen35` and install mlx-lm
make download   # fetch a model from Hugging Face (asks which)

make run PROMPT="Explain quantum entanglement in 3 sentences."
make chat
make bench
```

Each of these prints the model menu and waits for a number:

```
Choose a model (set MODEL=... to skip this prompt):
  1) Qwen3.5 9B   mlx-community/Qwen3.5-9B-4bit    6 GB    default
  2) Qwen3.5 4B   mlx-community/Qwen3.5-4B-4bit    2.5 GB  fastest, least memory
  3) Qwen3.8 27B  mlx-community/Qwen3.8-27B-4bit   16 GB   strongest, slowest
Model [1]:
```

Pass `MODEL=` to skip the prompt, for scripts or for a repo that is not in the menu.
Without a terminal (CI, piped stdin) the menu is skipped and the 9B is used:

```sh
make run MODEL=mlx-community/Qwen3.8-27B-4bit PROMPT="..."
make chat MODEL=mlx-community/Qwen3.5-9B-8bit
```

The menu lives in `scripts/choose-model.sh`; add a line there to offer another model.

Rust:

```sh
make rust-run PROMPT="Explain how a rainbow forms."
make rust-chat
make rust-chat2   # 4B model: faster, less memory
make rust-bench
```

The first Rust build compiles MLX from source and takes a few minutes.

## Make targets

| Target | Description |
|---|---|
| `setup` | Create the conda env and install dependencies |
| `download` | Download a model from Hugging Face |
| `run` / `chat` / `bench` | Python: one prompt, interactive chat, throughput |
| `rust-build` | Build the Rust runner |
| `rust-chat2` | Rust chat with `mlx-community/Qwen3.5-4B-4bit`, no prompt |
| `rust-run` / `rust-chat` / `rust-bench` | Rust: one prompt, interactive chat, throughput |
| `server` | OpenAI-compatible API on `127.0.0.1:8080` |
| `clean` / `clean-model` | Remove the conda env / a cached model |

All model-using targets show the model menu unless `MODEL=` is set.
Variables: `MODEL`, `PROMPT`, `MAX_TOKENS`, e.g. `make rust-chat MODEL=mlx-community/Qwen3.5-9B-8bit`.

## Rust runner options

```
rust/target/release/qwen35 [--model REPO_OR_DIR] [--prompt TEXT] [--max-tokens N]
                           [--temp T] [--top-k K] [--think] [--chat] [--bench]
                           [--no-kernel] [--print-ids]
```

- `--temp 0` is greedy decoding.
- `--think` enables Qwen's thinking mode (slower).
- `--no-kernel` uses the plain-ops recurrence instead of the Metal kernel (for verification).

## Layout

```
scripts/choose-model.sh  Model menu used by make
python/qwen.py      Python runner (mlx-lm)
rust/src/main.rs    CLI, model resolution, sampling, pipelined generation
rust/src/model.rs   Qwen3.5 text model: Gated DeltaNet + gated full attention layers
rust/src/kernel.rs  Gated DeltaNet recurrence as a custom Metal kernel
```

## License

MIT, see [LICENSE](LICENSE). The Rust model and kernel are derived from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
