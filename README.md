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

Quit with Ctrl-C or Ctrl-Z: Ctrl-Z is caught and shuts the example down instead of suspending
it, since a suspended run would keep the microphone and the model. `make audio` also kills any
older leftover run (`scripts/stop-audio.sh`) before starting. The Mac Studio and Mac mini have
no built-in microphone, so connect AirPods or a USB mic first.

```
Loaded mlx-community/whisper-large-v3-turbo in 0.9s
Microphone: AirPods Pro (1 ch, 24000 Hz, F32)
Listening (pause 700 ms ends an utterance, level > 0.010 counts as speech). Ctrl-C or Ctrl-Z to quit.
[voice 2.2 s, peak level 0.265]
USER: Turn on the lights in the kitchen.
[audio 2.2 s | encode 0.10 s | decode 8 tok @ 139 tok/s]
[voice 3.2 s, peak level 0.268]
USER: What is the weather like in Hong Kong tomorrow?
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

## Jarvis (voice assistant)

`make jarvis` runs a small assistant that lives on the Mac. It is its own crate, `jarvis/`, in a
Cargo workspace with `rust/`, which it uses as a library for Whisper and Qwen3.5.

- **Ears:** Whisper on MLX (Rust).
- **Brain:** Apple's on-device foundation model (Apple Intelligence), through one Swift helper
  process, `jarvis/swift/apple_ai.swift`, that keeps the conversation and also acts as the
  router below. `make jarvis BRAIN=mlx` thinks with Qwen3.5 on MLX in process instead. If
  Apple's model fails (macOS sometimes reports `SensitiveContentAnalysisML error 15` when one of
  its system models is unavailable), Jarvis loads Qwen3.5 on MLX on the spot and answers with
  that, still on the Mac; it tries Apple's model again on the next question.
- **Eyes:** ask about "my screen" and it takes a screenshot and asks Qwen3-VL on MLX
  (`jarvis/python/see.py`, mlx-vlm). Without the vision model, Apple's Vision framework reads
  the text and the brain answers from it.
- **Voice:** Kokoro on MLX (`jarvis/python/speak.py`, mlx-audio) for English and a few other
  languages; macOS `say` for the rest (Korean uses the Yuna voice). Each sentence is spoken as
  soon as it is written, while the rest of the reply is still being generated.
- **Full duplex, like talking to a person:** one thread only listens and transcribes, another
  thinks and speaks, so Jarvis hears you while it is thinking or talking. Say something and it
  stops at once and answers that instead; what the mic picks up of its own voice is recognised
  and dropped. `--half-duplex` mutes the mic until the reply is over, for speakers that leak
  badly into the microphone.
- **Router:** the same Apple model decides with a guided yes/no (a fresh session each time)
  whether a question needs live or recent information.
- **Cloud:** Grok (xAI) with web search answers only when you start a sentence with "AI, ..."
  ("AI, what is the weather tomorrow?") or when the router says the question needs live data,
  and only if `XAI_API_KEY` is set (`XAI_MODEL` picks the model, default `grok-4.7`).
  Everything else never leaves the Mac; asking "can you use cloud AI?" is answered on device
  and explains the rule. A searched answer is much dearer than a plain chat call.
- **Pictures:** "imagine a red apple on a table" makes one with Apple's Image Playground when
  the Mac supports it, otherwise with FLUX.1-schnell on MLX (`jarvis/python/imagine.py`,
  mflux, about 10 GB of weights on first use). The picture is saved under `~/Pictures/Jarvis`
  and drawn right in the terminal as a small 256 x 256 picture (`--image-size 512` for bigger)
  with the Kitty graphics protocol (kitty, WezTerm, Ghostty) or, in iTerm2, its own inline-image
  protocol (`--image-protocol kitty|iterm|none` to choose).
  While a picture is being made, a progress bar animates on the terminal
  (`🎨 ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ 2/3 steps · 6.4 s`); a cloud call shows a spinner with the
  elapsed time (`☁️  cloud ⠹ 4.2 s`).
  "describe ~/Desktop/photo.jpg" or any image URL shows the picture and has Apple Vision read
  it (text, scene labels, faces, animals, barcodes); the brain then answers your question
  about it.
- **Typing:** besides the microphone you can type requests into the terminal, which is the
  easy way to pass paths and URLs.
- **Which AI, local or cloud:** at startup a table lists every role (ears, brain, router, eyes,
  vision, imagine, voice, cloud) with local or CLOUD, model name and version; `--verbose` adds
  the path on disk and the size. Every call then prints one short line, e.g.
  `[ai local: Whisper (ears) · mlx-community/whisper-large-v3-turbo · rev a4aaeec]`
  or `[ai cloud: grok-4.7 via xAI with web search · api.x.ai/v1/responses (you asked for AI)]`.
  What you said is printed as `USER:`, local replies as `AI:`, Grok's as `CLOUD AI:`.
- **Threads:** the main thread only listens (microphone, level meter `🎤 ▁▂▅▇█ 1.4 s`, cutting
  utterances); a second thread transcribes with Whisper; a third thinks and speaks. So the
  microphone and interruptions stay responsive even while FLUX is drawing or Kokoro is talking.
  Buffer over/underruns while the machine is busy are reported once as a dropped bit of audio.
- **Natural speech:** replies are written as short spoken sentences, and markdown marks, URLs,
  bracketed notes and emoji are stripped before Kokoro reads them. Pick another Kokoro voice
  with `--voice` (af_heart, af_bella, am_adam, bf_emma, bm_george, ...).

```sh
make setup                                   # also installs mlx-vlm, mlx-audio, misaki[en]
make jarvis                                  # listen and answer (connect the AirPods first)
cd jarvis && make                            # the same, from the folder (`make help` lists the knobs)
make jarvis WAV=question.wav                 # replay a recording through the whole pipeline
make jarvis JARVIS_FLAGS="--no-speak"        # text only
make jarvis JARVIS_FLAGS="--local-only"      # never use the cloud
make jarvis WHISPER_LANG=ko                   # Korean in, Korean out
make jarvis BRAIN=mlx MODEL=mlx-community/Qwen3.5-4B-4bit   # Qwen3.5 on MLX as the brain
```

```
Models (local = on this Mac; --verbose for paths and sizes):
  ears     local  Whisper on MLX · mlx-community/whisper-large-v3-turbo · rev a4aaeec
  brain    local  Apple on-device model · Apple Foundation Models · macOS 27.0.1
  router   local  Apple on-device model · decides when a question needs the cloud
  eyes     local  Qwen3-VL on MLX · mlx-community/Qwen3-VL-4B-Instruct-4bit · mlx-vlm 0.7.8
  vision   local  Apple Vision framework · macOS 27.0.1 · text, labels, faces, animals, barcodes
  imagine  local  FLUX.1-schnell on MLX · mflux-community/flux-1-schnell-mflux-q4 · mflux 0.23.0 (Image Playground: notSupported)
  voice    local  Kokoro on MLX · voice af_heart · mlx-community/Kokoro-82M-bf16 · mlx-audio 0.5.9
  cloud    CLOUD  grok-4.7 via xAI with web search · only on "AI ..." or when the router asks
Full duplex: Jarvis keeps listening while it thinks and talks; just talk over it.
🎤 ▁▂▅▇█▆▃▂▁ 1.4 s
[ai local: Whisper (ears) · mlx-community/whisper-large-v3-turbo · rev a4aaeec]
USER: What is 2 plus 2?
[ai local: Apple on-device model (router): cloud needed = no]
[ai local: Apple on-device model (brain) · Apple Foundation Models · macOS 27.0.1]
AI: Two plus two is four.
[ai local: Kokoro on MLX (voice) · mlx-community/Kokoro-82M-bf16 · mlx-audio 0.5.9 · 1 sentence · 0.3 s]
USER: AI, what is the weather in Hong Kong tomorrow?
[ai cloud: grok-4.7 via xAI with web search · api.x.ai/v1/responses (you asked for AI)]
☁️  cloud ⠹ 4.2 s
CLOUD AI: Tomorrow in Hong Kong looks fine and dry, 26 to 32 degrees.
USER: imagine a red apple on a table
🎨 ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ 2/3 steps · 6.4 s
[ai local: FLUX.1-schnell on MLX (imagine) · mflux-community/flux-1-schnell-mflux-q4 · mflux 0.23.0 · 9.8 s]
Saved /Users/you/Pictures/Jarvis/1792316310-a-red-apple-on-a-table.png
AI: Here is your picture of a red apple on a table.
```

With no microphone attached Jarvis waits for one to appear (AirPods opening, the iPhone coming
near) instead of exiting, and reconnects if it goes away. While you talk, a live level meter runs on the terminal; it is replaced by the `USER:` transcript when you pause. Ctrl-C or Ctrl-Z quits at once, even mid-sentence: generation stops, playback is cut off, a
pending cloud call is cancelled and the helpers are closed.

Memory: Qwen3-VL-4B (3 GB), Whisper (1.6 GB) and Kokoro are resident; with `BRAIN=mlx` add the
Qwen3.5 model (6 GB for 9B).
The screenshot needs Screen Recording permission for your terminal (System Settings > Privacy &
Security); without it the eyes only see the wallpaper.

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
| `jarvis` | Rust: voice assistant (Whisper, Qwen3.5, Qwen3-VL, Kokoro on MLX; Grok on "AI ...") |
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

## Tests

```sh
make test          # unit tests, helpers, stop script, then end-to-end runs on synthesized speech
make test-quick    # the first three only
```

`scripts/test.sh` makes its recordings with `say`, so no microphone is needed, and speech is
written to files or played at volume 0. It checks transcription, on-device answers, echo
filtering, interruption, the screen path, typed requests, pictures (Apple Vision on a file and
on a URL, FLUX generation, Vision spotting a dog and a face in generated pictures), the cloud
policy (one Grok call when `XAI_API_KEY` is set, otherwise skipped), and that Ctrl-C and
Ctrl-Z mid-reply exit at once. The Python helpers have their own tests in
`jarvis/python/test_helpers.py`, which the suite runs as well.

## Layout

```
scripts/choose-model.sh  Model menu used by make
python/qwen.py      Python runner (mlx-lm)
rust/src/main.rs    CLI, model resolution, sampling, pipelined generation
rust/src/model.rs   Qwen3.5 text model: Gated DeltaNet + gated full attention layers
rust/src/kernel.rs  Gated DeltaNet recurrence as a custom Metal kernel
rust/src/lib.rs                 The qwen35 library: model, llm, hf, audio (used by the runner, the example and Jarvis)
rust/src/llm.rs                 Prompt format, sampling, pipelined decoding, multi-turn Chat with a persistent cache
rust/src/hf.rs                  Model files: local dir, Hugging Face cache, or `hf download`
rust/src/audio/listen.rs        Mic capture (cpal), VAD segmentation, WAV input, Ctrl-C/Ctrl-Z shutdown
rust/src/audio/stt.rs           Whisper transcriber
rust/src/audio/whisper.rs       Whisper encoder/decoder and greedy decoding on mlx-rs
rust/src/audio/mel.rs           Log-mel spectrogram (matches mlx-whisper bit for bit)
rust/src/audio/npy.rs           .npy/.npz weight loader
rust/examples/audio/main.rs     Audio agent CLI (microphone or WAV to text)
jarvis/Makefile                 `make`, `make WAV=…`, `make test`, `make stop` for the assistant alone
jarvis/src/main.rs              Jarvis: CLI, routing (brain / eyes / router / cloud), the listening loop
jarvis/src/brain.rs             Apple's model (default) or Qwen3.5 on MLX as the brain; the persona
jarvis/src/eyes.rs, voice.rs    Screenshot + Qwen3-VL; Kokoro or `say`
jarvis/src/pictures.rs, show.rs Making and reading pictures (Image Playground / FLUX, Apple Vision); Kitty/iTerm inline display
jarvis/src/apple.rs, cloud.rs   apple-ai helper client (router, OCR); Grok via the xAI Responses API
jarvis/python/see.py, speak.py, imagine.py  Persistent MLX helpers (mlx-vlm, mlx-audio, mflux), JSON lines over stdin/stdout
jarvis/swift/apple_ai.swift     Swift helper: FoundationModels chat/route, Vision analysis, Image Playground
rust/assets/mel_*.npy           Mel filterbanks
```

## License

MIT, see [LICENSE](LICENSE). The Rust model and kernel are derived from
[mlx-lm](https://github.com/ml-explore/mlx-lm), and the audio example from
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (both MIT, Apple Inc.);
see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
