1. Read @ralph/progress.txt
2. Run `bd ready` to decide which task to work on first. This should be the one YOU decide has the highest priority - not necessarily the first in the list.
3. Check any feedback loops, such as types and tests.
4. Append your progress to the ralph/progress.txt file.
5. Make a git commit of that feature.
6. Close the beads task.
7. Don't bother trying `git push`, you don't have credentials.

ONLY WORK ON A SINGLE FEATURE.

DO NOT USE PLAN MODE. For larger tasks that require a plan:
- Break it down into smaller tasks.
- Use `bd create` to create each smaller task.
- Use `bd close --reason "..."` to close the current large task. Mention in the reason that it was broken down into smaller tasks.