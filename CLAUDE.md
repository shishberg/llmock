# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until it is committed to git.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **git commit** - This is MANDATORY:
   ```bash
   git commit {changed files...} -m "{description...}"
   ```
5. **Clean up** - Clear stashes, etc.
6. **Verify** - All changes committed
7. **Hand off** - Provide context for next session

## Writing Tickets

Structure ticket descriptions to separate what's required from what's suggested:

- **Goal**: The functional requirement, motivation, or observation. What problem are we solving and why? This is the part that's set in stone — the agent must achieve this.
- **Suggested implementation**: A starting direction based on your current understanding of the code. Label this clearly as a suggestion. The agent may find a better approach once they're in the code, and that's fine — but having a concrete starting point saves time when it's right.
- **Context**: Relevant background the agent won't have — related recent changes, rejected alternatives and why, architectural constraints, or things you tried that didn't work. This helps the agent avoid dead ends.

The goal is to give the agent enough information to make good decisions without boxing them into a specific approach. When a ticket conflates requirements with implementation suggestions, agents tend to treat implementation details as hard requirements and spend effort forcing a suboptimal approach rather than stepping back to find a simpler one.

For tickets with strong interface contracts or compatibility requirements, say so explicitly in the goal section — don't rely on the agent inferring which parts are negotiable.

## Reading Tickets

When picking up a ticket, distinguish between:
- **Goal/requirements**: What must be true when you're done. Achieve this.
- **Suggested implementation**: The ticket author's best guess at how to do it, written without the code open. Use it as a starting point, but if you find a simpler way to achieve the goal, prefer that. If you diverge significantly, note why in your close reason.
- **Context**: Background to help you avoid dead ends. Don't ignore it.
