#!/bin/zsh
# End-to-end tests for the audio example, Inkling 2 Mini's image input and Jarvis: `make test`.
#
# Runs the Rust unit tests, then exercises the real binaries with recordings synthesized by
# `say`, so no microphone is needed and nothing is played aloud (speech goes to WAV files or
# plays at volume 0). Cloud checks run only when XAI_API_KEY is set (one Grok call, a few cents).
#
#   scripts/test.sh            everything
#   scripts/test.sh quick      unit tests, helpers and the stop script only (no model runs)
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
CONDA=${CONDA:-/opt/anaconda3/bin/conda}
ENV=${ENV:-qwen35}
PY=$($CONDA run --no-capture-output -n $ENV python -c 'import sys; print(sys.executable)' 2>/dev/null || echo python3)
# The binaries download a missing model with the `hf` CLI, which lives next to that python.
case $PY in */*) PATH="$PATH:$(dirname "$PY")" ;; esac
TMP=$(mktemp -d /tmp/jarvis-test.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
JARVIS=rust/target/release/jarvis
AUDIO=rust/target/release/examples/audio
APPLE_AI=rust/target/apple-ai
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "  ok   $1"; }
bad()  { fail=$((fail+1)); echo "  FAIL $1"; [ -n "${2:-}" ] && sed 's/^/       | /' <<< "$2" | head -20; }
check() { # check DESCRIPTION PATTERN TEXT  (PATTERN is a fixed string, case-insensitive)
    if grep -qiF -- "$2" <<< "$3"; then ok "$1"; else bad "$1 (missing: $2)" "$3"; fi
}
check_not() {
    if grep -qiF -- "$2" <<< "$3"; then bad "$1 (unexpected: $2)" "$3"; else ok "$1"; fi
}
say_wav() { say -o "$TMP/$1.wav" --data-format=LEI16@16000 "$2"; }
# say_cat OUT GAP_SECONDS PART...   concatenate recordings with silence between them
say_cat() {
    local out=$1 gap=$2; shift 2
    python3 -I - "$TMP" "$out" "$gap" "$@" <<'PY'
import wave, sys, os
d, out, gap, *parts = sys.argv[1:]
w_out = wave.open(os.path.join(d, out + ".wav"), "wb"); first = True
for n in parts:
    w = wave.open(os.path.join(d, n + ".wav"), "rb")
    if first: w_out.setparams(w.getparams()); first = False
    w_out.writeframes(w.readframes(w.getnframes()))
    w_out.writeframes(b"\x00" * int(w.getframerate() * 2 * float(gap)))
w_out.close()
PY
}
# run_jarvis WAV [flags...]  -> combined output; speech is captured to files, never played
run_jarvis() {
    local wav=$1; shift
    mkdir -p "$TMP/spoken"
    (cd jarvis && JARVIS_SPEAK_DIR="$TMP/spoken" "../$JARVIS" --wav "$wav" --apple-ai "../$APPLE_AI" --python "$PY" "$@" 2>&1)
}
# signal_jarvis WAV SIGNAL MARKER [flags...] -> "SECONDS EXIT STATE" after sending SIGNAL once MARKER appears
signal_jarvis() {
    local wav=$1 sig=$2 marker=$3; shift 3
    local log="$TMP/sig.log"
    (cd jarvis && "../$JARVIS" --wav "$wav" --apple-ai "../$APPLE_AI" --python "$PY" "$@" >"$log" 2>&1) &
    local pid=$!
    for i in $(seq 1 900); do grep -qF -- "$marker" "$log" && break; sleep 0.1; done
    local t0=$(date +%s.%N)
    kill -$sig $pid
    for i in $(seq 1 100); do kill -0 $pid 2>/dev/null || break; sleep 0.1; done
    local t1=$(date +%s.%N)
    local state=$(ps -o stat= -p $pid 2>/dev/null | tr -d ' ')
    kill -9 $pid 2>/dev/null
    wait $pid 2>/dev/null; local code=$?
    echo "$(printf '%.1f' $((t1-t0))) $code ${state:-gone}"
}
leftovers() { pgrep -fl 'target/apple-ai|see.py|speak.py|imagine.py|afplay|curl -sS' | grep -v pgrep || true; }

echo "== build"
cargo build --release --workspace --examples 2>&1 | grep -E "^(warning|error)" && bad "build has warnings/errors" || ok "workspace builds clean"
make -s -C jarvis apple-ai >/dev/null 2>&1 && ok "apple-ai helper builds" || bad "apple-ai helper build"

echo "== unit tests"
out=$(cargo test --release --workspace 2>&1)
check "cargo test passes" "test result: ok" "$out"
check_not "cargo test has no failures" "test result: FAILED" "$out"

echo "== stop script"
zsh -c 'exec -a rust/target/release/jarvis sleep 1000' & fake=$!; sleep 0.5; kill -STOP $fake
out=$(scripts/stop-audio.sh 2>&1)
check "stop-audio.sh finds a suspended jarvis" "suspended with Ctrl-Z" "$out"
kill -0 $fake 2>/dev/null && bad "stop-audio.sh killed it" || ok "stop-audio.sh killed it"
out=$(scripts/stop-audio.sh 2>&1); [ -z "$out" ] && ok "stop-audio.sh is quiet when idle" || bad "stop-audio.sh idle output" "$out"

echo "== apple-ai helper (Foundation Models + Vision)"
check "on-device model available" "available" "$($APPLE_AI status 2>&1)"
# macOS sometimes cannot run its own model ("SensitiveContentAnalysisML error 15"): that is a
# system fault, not ours. Jarvis then falls back to Qwen on MLX; checks that need Apple's model skip.
probe=$(echo '{"prompt":"Say OK."}' | $APPLE_AI chat 'Reply with one word.' 2>&1)
if grep -q '"done":true' <<< "$probe"; then apple_ok=1; else apple_ok=0; echo "  skip Apple's on-device model is failing system-wide right now: $(grep -o 'SensitiveContentAnalysisML error [0-9]*' <<< "$probe" | head -1)"; fi
if [ $apple_ok = 1 ]; then
check "router: general knowledge stays local" '"needs_cloud":false' "$(echo '{"route":"Who wrote Hamlet?"}' | $APPLE_AI chat)"
check "router: news goes to the cloud" '"needs_cloud":true' "$(echo '{"route":"What is the news today?"}' | $APPLE_AI chat)"
fi
printf 'Quarterly report 2026\nRevenue grew 12 percent.\n' > "$TMP/note.txt"
qlmanage -t -s 800 -o "$TMP" "$TMP/note.txt" >/dev/null 2>&1
check "Vision OCR reads rendered text" "Quarterly report" "$($APPLE_AI see "$TMP/note.txt.png" 2>&1)"
[ $apple_ok = 1 ] && check "chat streams and finishes" '"done":true' "$probe"
out=$($APPLE_AI see "$TMP/note.txt.png" 2>&1)
check "Vision reports faces/animals/barcodes fields" '"faces"' "$out"
out=$($APPLE_AI imagine "a red apple" "$TMP/playground.png" 2>&1)
if grep -q '"done":true' <<< "$out"; then ok "Image Playground generates a picture"
elif grep -q notSupported <<< "$out"; then echo "  skip Image Playground is not supported on this Mac (FLUX on MLX is used instead)"
else bad "Image Playground" "$out"; fi

echo "== python helpers (mlx-vlm, mlx-audio)"
out=$($PY -I jarvis/python/see.py --image "$TMP/note.txt.png" --prompt "What text is in this image?" 2>/dev/null)
check "see.py reads the image with Qwen3-VL" "12 percent" "$out"
out=$($PY -I jarvis/python/speak.py --text "Testing one two three." --wav "$TMP/tts.wav" 2>/dev/null)
check "speak.py synthesizes with Kokoro" '"done": true' "$out"
[ -s "$TMP/tts.wav" ] && [ $(stat -f %z "$TMP/tts.wav") -gt 20000 ] && ok "speak.py wrote a WAV" || bad "speak.py WAV missing or tiny"
if [ -d "$HOME/.cache/huggingface/hub/models--mflux-community--flux-1-schnell-mflux-q4" ]; then
    out=$(printf '%s\n%s\n' "{\"prompt\":\"a golden retriever dog sitting on grass, photo\",\"out\":\"$TMP/dog.png\",\"width\":512,\"height\":512,\"steps\":2,\"seed\":3}" \
        "{\"prompt\":\"close-up portrait photo of a smiling woman, natural light\",\"out\":\"$TMP/face.png\",\"width\":512,\"height\":512,\"steps\":2,\"seed\":5}" \
        | $PY -I jarvis/python/imagine.py 2>/dev/null)
    check "imagine.py generates with FLUX on MLX" '"done": true' "$out"
    [ -s "$TMP/dog.png" ] && [ -s "$TMP/face.png" ] && ok "imagine.py wrote the PNGs" || bad "imagine.py PNG missing"
    check "Vision recognises the animal in the generated picture" '"animals":["Dog"]' "$($APPLE_AI see "$TMP/dog.png" 2>&1)"
    out=$($APPLE_AI see "$TMP/face.png" 2>&1)
    if grep -qE '"faces":[1-9]' <<< "$out"; then ok "Vision finds the face in the generated portrait"; else bad "Vision found no face" "$out"; fi
else
    echo "  skip FLUX weights not downloaded (first 'imagine' downloads about 10 GB)"
fi
echo "== python helper unit tests"
out=$($PY -m unittest jarvis/python/test_helpers.py 2>&1)
check "python unittest passes" "OK" "$out"
check_not "python unittest has no failures" "FAILED" "$out"

echo "== make: IMAGE= goes to Inkling 2 Mini only"
out=$(make -n run MODEL=inkling-2-mini:26b-a5b-nvfp4 IMAGE=photo.jpg PROMPT=hi 2>&1)
check "make run passes IMAGE= to the Rust runner" '--image "photo.jpg" --prompt "hi"' "$out"
out=$(make -n chat MODEL=inkling-2-mini:26b-a5b-nvfp4 IMAGE=photo.jpg 2>&1)
check "make chat passes IMAGE= to the Rust runner" '--image "photo.jpg" --chat' "$out"
out=$(make run MODEL=mlx-community/Qwen3.5-4B-4bit IMAGE=photo.jpg 2>&1); code=$?
check "IMAGE= with a model that reads no images is refused" "does not read images" "$out"
[ $code != 0 ] && ok "and make fails" || bad "make succeeded with IMAGE= and a Qwen model" "$out"
check "the model menu says which model reads images" "reads images" "$(scripts/choose-model.sh --list)"

if [ "${1:-}" = quick ]; then echo; echo "quick: $pass passed, $fail failed"; exit $((fail > 0)); fi

# ---- Inkling 2 Mini: image input in the Rust runner (weights from Ollama's store) ----
INKLING=inkling-2-mini:26b-a5b-nvfp4
RUNNER=rust/target/release/qwen35
echo "== inkling: image input"
if [ -f "$HOME/.ollama/models/manifests/registry.ollama.ai/library/${INKLING%%:*}/${INKLING##*:}" ]; then
    # Two pictures drawn here: a page of text and a bar chart.
    $PY -I - "$TMP" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
font = lambda s: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)
im = Image.new("RGB", (800, 600), "white"); d = ImageDraw.Draw(im)
d.text((40, 30), "Quarterly report", fill="black", font=font(48))
for i, line in enumerate(["Revenue grew 12 percent to 4.8 million dollars.", "Costs fell 3 percent after the warehouse move.",
                          "Headcount: 42 people, 5 open positions.", "Next review: 14 January, room B204."]):
    d.text((40, 130 + 70 * i), line, fill=(20, 20, 60), font=font(30))
im.save(out + "/page.png")
im = Image.new("RGB", (1024, 768), (250, 250, 245)); d = ImageDraw.Draw(im)
d.text((60, 30), "Units sold by city, 2025", fill="black", font=font(40))
d.line((100, 660, 980, 660), fill="black", width=3); d.line((100, 120, 100, 660), fill="black", width=3)
for i, (name, v, col) in enumerate([("Seoul", 82, (200, 60, 60)), ("Tokyo", 64, (60, 120, 200)), ("Hong Kong", 97, (60, 170, 90)),
                                    ("Taipei", 35, (230, 160, 40)), ("Busan", 51, (140, 90, 190))]):
    x = 150 + i * 165
    d.rectangle((x, 660 - v * 5, x + 100, 660), fill=col)
    d.text((x + 30, 660 - v * 5 - 40), str(v), fill="black", font=font(28))
    d.text((x, 675), name, fill="black", font=font(26))
im.save(out + "/chart.png")
im.resize((3000, 2250)).save(out + "/big.jpg")
im = Image.new("RGB", (400, 200), (30, 60, 120)); d = ImageDraw.Draw(im)
d.text((40, 60), "PINEAPPLE 7341", fill="white", font=font(40))
im.save(out + "/word.png")
PY
    ollama stop $INKLING >/dev/null 2>&1   # two copies of the model do not fit comfortably
    out=$(INKLING_STATS=1 $RUNNER --model $INKLING --max-tokens 400 --image "$TMP/page.png" --prompt "Transcribe all the text in this image." 2>&1)
    check "the runner reads the text on a picture" "Revenue grew 12 percent to 4.8 million dollars" "$out"
    check "and the last line of it" "room B204" "$out"
    check "an 800x600 picture is 20x15 tokens" "[image 20x15 tokens" "$out"
    rust_gen=$(grep -o "gen [0-9]* tok @ [0-9.]*" <<< "$out" | tail -1 | awk '{print $5}')
    rust_prompt=$(awk '/^\[image/ { for (i = 1; i <= NF; i++) if ($i == "encode") e = $(i + 1) + 0 } /^\[prompt/ { p = $2 / $5 } END { printf "%.3f", e + p }' <<< "$out")

    out=$(make -s run MODEL=$INKLING IMAGE="$TMP/chart.png" PROMPT="Which city sold the most units, and how many?" 2>&1)
    check "make run IMAGE=... (menu entry 4) answers from a chart" "Hong Kong" "$out"
    check "with the number on its bar" "97" "$out"

    out=$(INKLING_STATS=1 $RUNNER --model $INKLING --max-tokens 200 --image "$TMP/big.jpg" --prompt "Which city sold the fewest units?" 2>&1)
    check "a 3000x2250 photo is scaled down to 26x20 tokens" "[image 26x20 tokens" "$out"
    check "and still read" "Taipei" "$out"

    # A chat: a mistyped file, a plain turn, two pictures in one message, then one more.
    out=$(printf '/image %s\n/image\nSay hello in three words.\n/image %s\n/image "%s"\nWhat is the title of the first picture, and which city leads in the second? Be brief.\n/image %s\nWhat number is written in this last picture?\n' \
        "$TMP/missing.png" "$TMP/page.png" "$TMP/chart.png" "$TMP/word.png" | make -s chat MODEL=$INKLING 2>&1)
    check "chat: a file that cannot be read is reported" "(not attached: cannot read image $TMP/missing.png" "$out"
    check "chat: /image without a file is reported" "(not attached: give a file" "$out"
    check "chat: two pictures go with one message" "(2 images attached to your next message)" "$out"
    check "chat: the first picture is read" "Quarterly report" "$out"
    check "chat: and the second" "Hong Kong" "$out"
    check "chat: a picture in a later turn is read" "7341" "$out"
    [ $(grep -c "^\[prompt " <<< "$out") = 3 ] && ok "chat: the conversation went on after the bad file (three turns)" || bad "chat: expected three turns" "$out"

    out=$($RUNNER --model $INKLING --image "$TMP/missing.png" --prompt "What is this?" 2>&1); code=$?
    check "a missing --image file is an error" "cannot read image" "$out"
    [ $code != 0 ] && ok "and the runner fails" || bad "the runner succeeded without its image"

    # The same picture and question through Ollama itself: the Rust runner must not be slower.
    if curl -s -m 3 localhost:11434/api/version >/dev/null 2>&1; then
        speed() { # -> "PROMPT_SECONDS GEN_TOK_PER_S" for the page picture, greedy, thinking effort 0
            python3 -I - "$TMP/page.png" <<'PY'
import sys, json, base64, urllib.request
body = {"model": "inkling-2-mini:26b-a5b-nvfp4", "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 400},
        "messages": [{"role": "user", "content": "Transcribe all the text in this image.", "images": [base64.b64encode(open(sys.argv[1], "rb").read()).decode()]}]}
d = json.loads(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})).read())
print("%.3f %.1f" % (d["prompt_eval_duration"] / 1e9, d["eval_count"] / (d["eval_duration"] / 1e9)))
PY
        }
        # Load the model with a text prompt: Ollama keeps the cache of a prompt it has read, so
        # the picture must be new to it when it is timed (the model was stopped above).
        printf '{"model":"%s","prompt":"hi","stream":false,"think":false,"options":{"num_predict":1}}' $INKLING | curl -s -m 300 localhost:11434/api/generate -d @- >/dev/null
        r=($(speed 2>/dev/null))
        ollama stop $INKLING >/dev/null 2>&1
        # The Rust runner again, now that the weights are back in the file cache (its first
        # run above came straight after Ollama's copy was unloaded): the better of the two counts.
        out=$(INKLING_STATS=1 $RUNNER --model $INKLING --max-tokens 400 --image "$TMP/page.png" --prompt "Transcribe all the text in this image." 2>&1)
        gen2=$(grep -o "gen [0-9]* tok @ [0-9.]*" <<< "$out" | tail -1 | awk '{print $5}')
        prompt2=$(awk '/^\[image/ { for (i = 1; i <= NF; i++) if ($i == "encode") e = $(i + 1) + 0 } /^\[prompt/ { p = $2 / $5 } END { printf "%.3f", e + p }' <<< "$out")
        (( ${gen2:-0} > rust_gen )) && rust_gen=$gen2
        (( ${prompt2:-9} < rust_prompt )) && rust_prompt=$prompt2
        if [ ${#r} = 2 ]; then
            echo "       picture and prompt read in: Rust ${rust_prompt}s, Ollama ${r[1]}s; generation: Rust ${rust_gen} tok/s, Ollama ${r[2]} tok/s"
            (( rust_gen > r[2] )) && ok "generation after a picture is faster than Ollama's (${rust_gen} vs ${r[2]} tok/s)" || bad "generation after a picture is slower than Ollama's (${rust_gen} vs ${r[2]} tok/s)"
            (( rust_prompt < r[1] * 1.1 )) && ok "the picture and prompt are read as fast as Ollama does, within 10% (${rust_prompt}s vs ${r[1]}s)" || bad "the picture and prompt are read more than 10% slower than Ollama does (${rust_prompt}s vs ${r[1]}s)"
        else
            echo "  skip Ollama did not answer: speed comparison skipped"
        fi
    else
        echo "  skip Ollama is not running: speed comparison skipped"
    fi
else
    echo "  skip $INKLING is not in Ollama's store (make ollama-pull)"
fi

echo "== recordings"
say_wav q1 "What is two plus two?"
say_wav echo "Two plus two equals four."
say_wav screen "What is on my screen right now?"
say_wav ai "A I, what is the weather in Hong Kong tomorrow?"
say_wav story "Tell me a long story about a robot who learns to paint."
say_cat duplex 1.0 q1 echo screen
ok "synthesized with say"

echo "== audio example"
out=$($AUDIO --wav "$TMP/q1.wav" 2>&1)
check "audio example transcribes" "2 plus 2" "$out"
check "audio example reports timing" "[audio" "$out"

echo "== jarvis: local pipeline, full duplex (volume 0)"
out=$(cd jarvis && "../$JARVIS" --wav "$TMP/duplex.wav" --apple-ai "../$APPLE_AI" --python "$PY" --local-only --volume 0 2>&1)
check "transcribes the question" "2 plus 2" "$out"
check "answers on device" "AI: " "$out"
check "speaks with Kokoro" "[ai local: Kokoro on MLX (voice)" "$out"
check "drops its own voice coming back" "[ignored: that was my own voice]" "$out"
check "interrupts speech when spoken to" "[interrupted]" "$out"
check "looks at the screen with Qwen3-VL" "[ai local: Qwen3-VL on MLX (eyes)" "$out"
check_not "local-only never calls the cloud" "[ai cloud" "$out"
check "every answer names its AI and version" "[ai local: Apple on-device model (brain) · Apple Foundation Models · macOS" "$out"
check "transcription names its AI and version" "[ai local: Whisper (ears) · mlx-community/whisper-large-v3-turbo · rev" "$out"
check_not "per-call lines stay short (no paths)" "(ears) · mlx-community/whisper-large-v3-turbo · rev a4aaeec · ~" "$out"
check "startup lists the models" "Models (local = on this Mac" "$out"
check "startup names the ears" "  ears     local  Whisper on MLX" "$out"
check "startup names the image model" "  imagine  local  " "$out"
check "startup says the cloud is off in local-only" "  cloud    -      off" "$out"
check "one apple-ai process (brain + router)" "  brain    local  Apple on-device model" "$out"
check "startup lists the voice model with its facts" "  voice    local  Kokoro on MLX · voice af_heart · mlx-community/Kokoro-82M-bf16 · mlx-audio" "$out"
check_not "cached models are not reported as missing" "not downloaded yet" "$out"
out=$(cd jarvis && "../$JARVIS" --wav "$TMP/q1.wav" --no-speak --no-eyes --no-imagine --local-only --verbose --apple-ai "../$APPLE_AI" --python "$PY" 2>&1)
check "--verbose shows the path and size at startup" "  ears     local  Whisper on MLX · mlx-community/whisper-large-v3-turbo · rev a4aaeec · ~/.cache/huggingface/hub/" "$out"
check "--verbose shows the size" " · 1.6 GB" "$out"
[ -z "$(leftovers)" ] && ok "no helper processes left" || bad "helper processes left" "$(leftovers)"

echo "== jarvis: typed requests, pictures (Apple Vision), image generation"
out=$(cd jarvis && printf 'describe %s\nimagine a red apple on a table\n' "$TMP/note.txt.png" \
    | JARVIS_PICTURES="$TMP/pictures" "../$JARVIS" --stdin --no-speak --no-eyes --local-only --image-protocol none \
      --apple-ai "../$APPLE_AI" --python "$PY" 2>&1)
check "typed request is echoed" "USER: describe" "$out"
check "Apple Vision read the picture" "[ai local: Apple Vision (picture) · Vision framework · macOS" "$out"
check "the brain answered about the picture" "12 percent" "$out"
check "imagine request recognised" "[imagining: a red apple on a table" "$out"
if grep -qF "[ai local: FLUX.1-schnell on MLX (imagine)" <<< "$out" || grep -qF "[ai local: Image Playground (imagine)" <<< "$out"; then ok "a picture was generated"; else bad "no picture generated" "$out"; fi
check "picture saved" "Saved $TMP/pictures/" "$out"
check "a progress bar animated the generation" "3/3 steps" "$out"
check "end of typed input ends Jarvis" "End of input: shutting down." "$out"
[ -z "$(leftovers)" ] && ok "no helper processes left" || bad "helper processes left" "$(leftovers)"
# A picture by URL: serve the test directory over local HTTP.
python3 -u -m http.server 0 --bind 127.0.0.1 --directory "$TMP" >"$TMP/http.log" 2>/dev/null & httpd=$!
for i in $(seq 1 50); do grep -q "port" "$TMP/http.log" && break; sleep 0.1; done
port=$(grep -o "port [0-9]*" "$TMP/http.log" | head -1 | cut -d' ' -f2)
out=$(cd jarvis && printf 'what does http://127.0.0.1:%s/note.txt.png say?\n' "$port" \
    | "../$JARVIS" --stdin --no-speak --no-eyes --no-imagine --local-only --image-protocol none --apple-ai "../$APPLE_AI" --python "$PY" 2>&1)
kill $httpd 2>/dev/null; wait $httpd 2>/dev/null
check "a picture by URL is downloaded and read" "[vision: Image 800x800" "$out"
check "the brain answers about the URL picture" "12 percent" "$out"

echo "== jarvis: cloud policy"
out=$(unset XAI_API_KEY; run_jarvis "$TMP/ai.wav" --no-eyes --no-speak)
check "without a key, 'AI ...' stays on device" "no XAI_API_KEY is set: answering on device" "$out"
check_not "without a key, no cloud call" "[ai cloud" "$out"
if [ -n "${XAI_API_KEY:-}" ]; then
    out=$(run_jarvis "$TMP/ai.wav" --no-eyes --no-speak)
    check "'AI ...' goes to Grok" "[ai cloud: grok-4.7 via xAI" "$out"
check "the cloud call says why and where" "api.x.ai/v1/responses (you asked for AI)" "$out"
check "a spinner animated the cloud call" "☁️  cloud " "$out"
    check "Grok answers" "CLOUD AI: " "$out"
check "startup shows the cloud model" "  cloud    CLOUD  grok-4.7 via xAI" "$out"
    out=$(run_jarvis "$TMP/q1.wav" --no-eyes --no-speak)
    check_not "a plain question stays local even with a key" "[ai cloud" "$out"
[ $apple_ok = 1 ] && check "the router reported its decision" "(router): cloud needed = no" "$out"
else
    echo "  skip XAI_API_KEY not set: Grok checks skipped"
fi

echo "== jarvis: when the Apple model fails, Qwen on MLX answers instead"
out=$(cd jarvis && JARVIS_TEST_FAIL_BRAIN=1 "../$JARVIS" --wav "$TMP/q1.wav" --no-speak --no-eyes --no-imagine --local-only --apple-ai "../$APPLE_AI" --python "$PY" 2>&1)
check "the fallback is announced" "loading Qwen3.5 on MLX as the local fallback" "$out"
check "the fallback brain names itself" "[ai local: Qwen3.5 on MLX (brain, fallback) · mlx-community/Qwen3.5-9B-4bit" "$out"
check "the fallback answers" "AI: " "$out"
[ -z "$(leftovers)" ] && ok "no helper processes left" || bad "helper processes left" "$(leftovers)"
# With speech on, Qwen must not share the GPU with Kokoro while generating: speech is deferred.
out=$(cd jarvis && JARVIS_TEST_FAIL_BRAIN=1 JARVIS_SPEAK_DIR="$TMP/spoken-fb" "../$JARVIS" --wav "$TMP/q1.wav" --no-eyes --no-imagine --local-only --apple-ai "../$APPLE_AI" --python "$PY" 2>&1)
tps=$(grep -o "tok @ [0-9]* tok/s" <<< "$out" | tail -1 | grep -o "@ [0-9]*" | tr -dc 0-9)
[ -n "$tps" ] && [ "$tps" -ge 30 ] && ok "Qwen keeps its speed with speech on ($tps tok/s)" || bad "Qwen slowed down with speech on (${tps:-?} tok/s)" "$out"
check "the fallback reply was spoken" "[ai local: Kokoro on MLX (voice)" "$out"

echo "== jarvis: a dead model thread ends the program"
t0=$(date +%s)
out=$(cd jarvis && JARVIS_TEST_PANIC=ears "../$JARVIS" --wav "$TMP/q1.wav" --no-speak --no-eyes --no-imagine --local-only --apple-ai "../$APPLE_AI" --python "$PY" 2>&1)
check "the dead thread is reported" "[ears: the transcription thread stopped; shutting down]" "$out"
(( $(date +%s) - t0 < 20 )) && ok "Jarvis exited within 20 s" || bad "Jarvis took too long to exit after a thread died"
[ -z "$(leftovers)" ] && ok "no helper processes left" || bad "helper processes left" "$(leftovers)"

echo "== jarvis: Ctrl-C / Ctrl-Z"
r=($(signal_jarvis "$TMP/story.wav" INT "AI:" --no-speak --no-eyes --local-only))
[ "${r[2]}" = 0 ] && [ "${r[3]}" = gone ] && (( ${r[1]} < 5 )) && ok "Ctrl-C mid-reply exits cleanly in ${r[1]} s" || bad "Ctrl-C: ${r[*]}" "$(cat "$TMP/sig.log")"
r=($(signal_jarvis "$TMP/story.wav" TSTP "AI:" --no-speak --no-eyes --local-only))
[ "${r[2]}" = 0 ] && [ "${r[3]}" = gone ] && (( ${r[1]} < 5 )) && ok "Ctrl-Z mid-reply exits instead of suspending (${r[1]} s)" || bad "Ctrl-Z: ${r[*]}" "$(cat "$TMP/sig.log")"
[ -z "$(leftovers)" ] && ok "no helper processes left" || bad "helper processes left" "$(leftovers)"

echo; echo "$pass passed, $fail failed"
exit $((fail > 0))
