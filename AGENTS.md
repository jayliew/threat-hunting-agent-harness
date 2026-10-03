# Agent instructions

These instructions apply to any coding agent working in this repository.

## Git workflow

`dev` is the primary branch. Completed work from other branches and git worktrees is merged into `dev` by pull request. A pull request or merge from `dev` to `main` is manual and done by a person only. Agents never open that pull request and never merge `dev` into `main`.

Do not commit on `dev` or `main`.

1. Fetch the latest remotes.
2. Start from `dev`. If `dev` is missing locally, check out `origin/dev`. If it does not exist on the remote, create it from `main` and push it.
3. Create a **new branch** from `dev` for this task, in this checkout or in a worktree. Name it for the change.
4. Do the work only on that branch. Keep the branch focused on the requested change.
5. When the work is done, push the branch and open a **pull request that targets `dev`**. Assume every agent change is merged to `dev` with a pull request, not to `main`.

Do not merge that pull request unless you are asked to.
