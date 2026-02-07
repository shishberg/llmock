#!/bin/bash
# Usage: ./loop.sh [plan] [max_iterations]
# Examples:
#   ./loop.sh              # Build mode, unlimited iterations
#   ./loop.sh 20           # Build mode, max 20 iterations
#   ./loop.sh plan         # Plan mode, unlimited iterations
#   ./loop.sh plan 5       # Plan mode, max 5 iterations

DIR="$(dirname "$0")"
pwd

# Parse arguments
if [ "$1" = "plan" ]; then
    # Plan mode
    # MODE="plan"
    # PROMPT_FILE="PROMPT_plan.md"
    # MAX_ITERATIONS=${2:-0}
    echo "Plan mode not working yet"
    exit 1
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    # Build mode with max iterations
    MODE="build"
    PROMPT_FILE="$DIR/PROMPT_build.md"
    MAX_ITERATIONS=$1
else
    # Build mode, unlimited (no arguments or invalid input)
    MODE="build"
    PROMPT_FILE="$DIR/PROMPT_build.md"
    MAX_ITERATIONS=0
fi

ITERATION=0
CURRENT_BRANCH=$(git branch --show-current)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode:   $MODE"
echo "Prompt: $PROMPT_FILE"
echo "Branch: $CURRENT_BRANCH"
[ $MAX_ITERATIONS -gt 0 ] && echo "Max:    $MAX_ITERATIONS iterations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

PREV_READY=""
STALE_COUNT=0

while true; do
    if [ $MAX_ITERATIONS -gt 0 ] && [ $ITERATION -ge $MAX_ITERATIONS ]; then
        echo "Reached max iterations: $MAX_ITERATIONS"
        break
    fi

    # Run Ralph iteration with selected prompt
    cat "$PROMPT_FILE" | claude -p \
        --dangerously-skip-permissions \
        --output-format=stream-json \
        --verbose \
    | jq --unbuffered -r '
        # Pastel ANSI colors (256-color mode)
        def lavender: "\u001b[38;5;183m";
        def blue:     "\u001b[38;5;117m";
        def green:    "\u001b[38;5;157m";
        def yellow:   "\u001b[38;5;229m";
        def pink:     "\u001b[38;5;218m";
        def dim:      "\u001b[2m";
        def bold:     "\u001b[1m";
        def reset:    "\u001b[0m";

        if .type == "assistant" then
            .message.content[]? |
            if .type == "text" and (.text | test("\\S")) then
                lavender + (.text | gsub("^\\s+|\\s+$"; "")) + reset
            elif .type == "tool_use" then
                if .name == "AskUserQuestion" then
                    "\n" + pink + bold + "  [QUESTION] " + reset + pink +
                    (.input.questions // [] | map(.question) | join("\n             ")) +
                    reset + "\n"
                else
                    blue + "  [" + .name + "] " + reset + dim +
                    (if   .name == "Read"      then (.input.file_path // "")
                     elif .name == "Write"     then (.input.file_path // "")
                     elif .name == "Edit"      then (.input.file_path // "")
                     elif .name == "Glob"      then (.input.pattern // "")
                     elif .name == "Grep"      then (.input.pattern // "") + " in " + (.input.path // ".")
                     elif .name == "Bash"      then "$ " + ((.input.command // "") | .[0:120])
                     elif .name == "WebSearch" then "\"" + (.input.query // "") + "\""
                     elif .name == "WebFetch"  then (.input.url // "")
                     elif .name == "Task"      then (.input.description // "")
                     else ((.input // {}) | tostring)
                     end) + reset
                end
            else empty
            end
        elif .type == "user" then
            .message.content[]? |
            if .type == "tool_result" then
                green + dim + "    <- " +
                ((.content // "") | tostring | length | tostring) + " chars" + reset
            else empty
            end
        elif .type == "result" then
            "\n" + yellow + bold + "  [DONE] " + reset +
            yellow + dim + ((.result // "") | tostring | .[0:300]) + reset + "\n"
        else empty
        end
    '

    # Check remaining tasks
    CURRENT_READY=$(bd ready --json 2>/dev/null)

    if [ "$CURRENT_READY" = "[]" ] || [ -z "$CURRENT_READY" ]; then
        echo -e "\n\033[38;5;229m\033[1mNo tasks remaining, exiting.\033[0m"
        break
    fi

    # Detect stuck loop: break after N consecutive identical task lists
    if [ "$CURRENT_READY" = "$PREV_READY" ]; then
        STALE_COUNT=$((STALE_COUNT + 1))
        echo -e "\033[38;5;210m\033[2m  (task list unchanged: $STALE_COUNT)\033[0m"
        if [ $STALE_COUNT -ge 1 ]; then
            echo -e "\n\033[38;5;210m\033[1mTask list unchanged, exiting.\033[0m"
            break
        fi
    else
        STALE_COUNT=0
    fi
    PREV_READY="$CURRENT_READY"

    # # Push changes after each iteration
    # git push origin "$CURRENT_BRANCH" || {
    #     echo "Failed to push. Creating remote branch..."
    #     git push -u origin "$CURRENT_BRANCH"
    # }

    ITERATION=$((ITERATION + 1))
    echo -e "\n\n======================== LOOP $ITERATION ========================\n"
done
