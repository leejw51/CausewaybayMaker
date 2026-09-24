# CausewaybayMaker

Run [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) locally on Apple Silicon with
[MLX](https://github.com/ml-explore/mlx), from Python or Rust.

Uses the MLX 4-bit build [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit),
downloaded from Hugging Face on first use.

| Runner | Generation | Prompt | Memory |
|---|---|---|---|
| Rust (`mlx-rs`) | ~75–82 tok/s | ~420 tok/s | 5.2 GB |
| Python (`mlx-lm`) | ~72–90 tok/s | ~330 tok/s | 5.2 GB |

Measured on an M5 Max. The 8-bit model (`mlx-community/Qwen3.5-9B-8bit`) runs at ~37 tok/s and uses 9.7 GB.

## Requirements

- Apple Silicon Mac
- [Anaconda](https://www.anaconda.com/) for the Python runner
- Rust toolchain and `cmake` (`brew install cmake`) for the Rust runner

## Quick start

```sh
make setup      # create conda env `qwen35` and install mlx-lm
make download   # fetch the model from Hugging Face

make run PROMPT="Explain quantum entanglement in 3 sentences."
make chat
make bench
```

Rust:

```sh
make rust-run PROMPT="Explain how a rainbow forms."
make rust-chat
make rust-bench
```

The first Rust build compiles MLX from source and takes a few minutes.

## Make targets

| Target | Description |
|---|---|
| `setup` | Create the conda env and install dependencies |
| `download` | Download the model from Hugging Face |
| `run` / `chat` / `bench` | Python: one prompt, interactive chat, throughput |
| `rust-build` | Build the Rust runner |
| `rust-run` / `rust-chat` / `rust-bench` | Rust: one prompt, interactive chat, throughput |
| `server` | OpenAI-compatible API on `127.0.0.1:8080` |
| `clean` / `clean-model` | Remove the conda env / the cached model |

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
python/qwen.py      Python runner (mlx-lm)
rust/src/main.rs    CLI, model resolution, sampling, pipelined generation
rust/src/model.rs   Qwen3.5 text model: Gated DeltaNet + gated full attention layers
rust/src/kernel.rs  Gated DeltaNet recurrence as a custom Metal kernel
```

## License

MIT, see [LICENSE](LICENSE). The Rust model and kernel are derived from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
