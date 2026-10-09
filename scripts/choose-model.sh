#!/bin/sh
# Interactive model picker. Prints the chosen Hugging Face repo id on stdout;
# the menu itself goes to stderr so the caller can capture the result.
#
#   scripts/choose-model.sh          # prompt (Enter = first entry)
#   scripts/choose-model.sh --list   # print the table and exit
#
# The list below is the single source of truth for the models offered by `make`.
MODELS='
mlx-community/Qwen3.5-9B-4bit   | Qwen3.5 9B  | 6 GB   | default
mlx-community/Qwen3.5-4B-4bit   | Qwen3.5 4B  | 2.5 GB | fastest, least memory
mlx-community/Qwen3.8-27B-4bit  | Qwen3.8 27B | 16 GB  | strongest, slowest
'

rows() { printf '%s\n' "$MODELS" | sed -e '/^[[:space:]]*$/d' -e 's/ *| */|/g'; }
repo() { rows | sed -n "${1}p" | cut -d'|' -f1; }

list() {
    i=0
    rows | while IFS='|' read -r r name size note; do
        i=$((i + 1))
        printf '  %d) %-12s %-32s %-7s %s\n' "$i" "$name" "$r" "$size" "$note"
    done
}

if [ "$1" = "--list" ]; then
    list
    exit 0
fi

n=$(rows | wc -l | tr -d ' ')
if ! ( : < /dev/tty ) 2>/dev/null; then
    # Not interactive (CI, piped): use the default silently.
    repo 1
    exit 0
fi

{
    echo "Choose a model (set MODEL=... to skip this prompt):"
    list
} >&2
while :; do
    printf 'Model [1]: ' >&2
    read -r choice < /dev/tty || exit 1
    [ -z "$choice" ] && choice=1
    case "$choice" in
        *[!0-9]*) ;;
        *) if [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then repo "$choice"; exit 0; fi ;;
    esac
    echo "Enter a number between 1 and $n." >&2
done
