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

## Audio agent (microphone to text)

`make audio` is a Rust example that listens on the microphone continuously and
transcribes with [Whisper](https://github.com/openai/whisper) running on MLX. The model is a
port of Apple's [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
to `mlx-rs`, using the `mlx-community/whisper-*` weights.

```sh
make audio                       # listen; every time you pause, the utterance is transcribed
make audio DEVICE=AirPods        # choose the input device by (part of) its name
make audio2                      # shortcut for the line above
make audio WAV=clip.wav          # transcribe a WAV file (any rate/channels)
make audio WHISPER=mlx-community/whisper-base-mlx WHISPER_LANG=ko
```

```
Loaded mlx-community/whisper-large-v3-turbo in 0.9s
Microphone: AirPods Pro (1 ch, 24000 Hz, F32)
Listening (pause 700 ms ends an utterance, level > 0.010 counts as speech). Ctrl-C to quit.
[voice 2.2 s, peak level 0.265]
>>>  Turn on the lights in the kitchen.
[audio 2.2 s | encode 0.10 s | decode 8 tok @ 139 tok/s]
[voice 3.2 s, peak level 0.268]
>>>  What is the weather like in Hong Kong tomorrow?
```

The mic stream never stops: a small energy-based voice activity detector cuts
the audio into utterances at pauses, and each one is transcribed while the next is
being recorded. Utterances are capped at 30 s (one Whisper window). Tune with the
example's own flags (`cd rust && cargo run --release --example audio -- --help`):
`--silence-ms` (pause length), `--vad-threshold` (raise it if background noise
triggers transcripts; the `peak level` line shows what your speech measures),
`--min-speech-ms`, `--list-devices`.

| Model | Size | Notes |
|---|---|---|
| `mlx-community/whisper-large-v3-turbo` (default) | 1.6 GB | best accuracy, ~0.1 s encode on M5 Max |
| `mlx-community/whisper-small-mlx` | 0.5 GB | |
| `mlx-community/whisper-base-mlx` | 0.15 GB | smallest, least accurate |

Any `mlx-community/whisper-*` repo works (`weights.safetensors` or `weights.npz`).
The tokenizer is taken from the matching `openai/whisper-*` repo. On first use
macOS asks for microphone permission for your terminal.

## Make targets

| Target | Description |
|---|---|
| `setup` | Create the conda env and install dependencies |
| `download` | Download a model from Hugging Face |
| `run` / `chat` / `bench` | Python: one prompt, interactive chat, throughput |
| `rust-build` | Build the Rust runner |
| `rust-chat2` | Rust chat with `mlx-community/Qwen3.5-4B-4bit`, no prompt |
| `rust-run` / `rust-chat` / `rust-bench` | Rust: one prompt, interactive chat, throughput |
| `audio` | Rust: live microphone (or `WAV=`) to text with Whisper on MLX |
| `audio2` | `make audio DEVICE=AirPods` |
| `server` | OpenAI-compatible API on `127.0.0.1:8080` |
| `clean` / `clean-model` | Remove the conda env / a cached model |

All model-using targets show the model menu unless `MODEL=` is set.
Variables: `MODEL`, `PROMPT`, `MAX_TOKENS`, e.g. `make rust-chat MODEL=mlx-community/Qwen3.5-9B-8bit`.
`make audio` uses `WHISPER`, `WHISPER_LANG`, `DEVICE`, `WAV` instead.

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
rust/examples/audio/main.rs     Audio agent: CLI, live mic capture (cpal), VAD segmentation, WAV input
rust/examples/audio/whisper.rs  Whisper encoder/decoder and greedy decoding on mlx-rs
rust/examples/audio/mel.rs      Log-mel spectrogram (matches mlx-whisper bit for bit)
rust/examples/audio/npy.rs      .npy/.npz weight loader
rust/assets/mel_*.npy           Mel filterbanks
```

## License

MIT, see [LICENSE](LICENSE). The Rust model and kernel are derived from
[mlx-lm](https://github.com/ml-explore/mlx-lm), and the audio example from
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (both MIT, Apple Inc.);
see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
