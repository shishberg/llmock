@ralph/progress.txt
1. Run `bd ready` to decide which task to work on first. This should be the one YOU decide has the highest priority - not necessarily the first in the list.
2. Check any feedback loops, such as types and tests.
3. Append your progress to the ralph/progress.txt file.
4. Make a git commit of that feature.
5. Close the beads task.
6. Don't bother trying `git push`, you don't have credentials.

ONLY WORK ON A SINGLE FEATURE.

DO NOT USE PLAN MODE. For larger tasks that require a plan:
- Break it down into smaller tasks.
- Use `bd create` to create each smaller task.
- Use `bd close --reason "..."` to close the current large task. Mention in the reason that it was broken down into smaller tasks.