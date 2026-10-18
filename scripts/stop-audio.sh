#!/bin/sh
# Stop leftover runs of the audio example or Jarvis before starting a new one.
#
# A run that was suspended with Ctrl-Z (instead of ended with Ctrl-C) keeps the
# microphone stream and the Whisper model open while sitting stopped in the
# background. `make audio` calls this first so a fresh run gets the mic.
#
# Only the example binary itself (debug or release) qualifies: the executable
# must be .../examples/audio, not just a shell whose command line mentions it.
# The [a] keeps this script's own command line from matching the pattern.
PATTERN='target/(release|debug)/(examples/[a]udio|[j]arvis)( |$)'
audio_pids() {
    for pid in $(pgrep -f "$PATTERN"); do
        case "$(ps -o comm= -p "$pid")" in
            */examples/audio | */jarvis) echo "$pid" ;;
        esac
    done
}

pids=$(audio_pids)
[ -z "$pids" ] && exit 0

for pid in $pids; do
    state=$(ps -o stat= -p "$pid" | tr -d ' ')
    case "$state" in
        T*) what="suspended with Ctrl-Z" ;;
        *)  what="still running" ;;
    esac
    echo "Killing old audio run (pid $pid, $what): $(ps -o command= -p "$pid")" >&2
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    # SIGKILL: a stopped process never gets to handle SIGTERM.
    kill -9 "$pid" 2>/dev/null
    # Its make/sh parents were suspended too; let them resume so they notice and exit.
    [ -n "$pgid" ] && kill -CONT -- "-$pgid" 2>/dev/null
done

# Wait briefly until they are gone so the mic is free before the new run opens it.
i=0
while [ $i -lt 20 ] && [ -n "$(audio_pids)" ]; do
    sleep 0.1
    i=$((i + 1))
done
exit 0
