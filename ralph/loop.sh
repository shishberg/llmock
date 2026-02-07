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

JQ_OUTPUT=$(mktemp)
trap 'rm -f "$JQ_OUTPUT"' EXIT

while true; do
    if [ $MAX_ITERATIONS -gt 0 ] && [ $ITERATION -ge $MAX_ITERATIONS ]; then
        echo "Reached max iterations: $MAX_ITERATIONS"
        break
    fi

    # Run Ralph iteration with selected prompt
    # jq extracts interesting info and emits __RALPH_COMPLETE__ if the
    # assistant (not the prompt) outputs <promise>COMPLETE</promise>
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
                lavender + (.text | gsub("^\\s+|\\s+$"; "")) + reset,
                if (.text | test("<promise>COMPLETE</promise>")) then
                    "__RALPH_COMPLETE__"
                else empty
                end
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
    ' | tee "$JQ_OUTPUT"

    # Check jq's formatted output for the sentinel (only emitted for assistant messages)
    if grep -q '__RALPH_COMPLETE__' "$JQ_OUTPUT"; then
        echo -e "\n\033[38;5;229m\033[1mAll work complete, exiting.\033[0m"
        break
    fi

    # # Push changes after each iteration
    # git push origin "$CURRENT_BRANCH" || {
    #     echo "Failed to push. Creating remote branch..."
    #     git push -u origin "$CURRENT_BRANCH"
    # }

    ITERATION=$((ITERATION + 1))
    echo -e "\n\n======================== LOOP $ITERATION ========================\n"
done
