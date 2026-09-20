# Agent instructions

These instructions apply to any coding agent working in this repository.

## Git workflow

Do not commit on `dev` or `main`.

1. Fetch the latest remotes.
2. Start from `dev`. If `dev` is missing locally, check out `origin/dev`. If it does not exist on the remote, create it from `main` and push it.
3. Create a **new branch** from `dev` for this task. Name it for the change.
4. Do the work only on that branch. Keep the branch focused on the requested change.
5. When the work is done, push the branch and open a **pull request that targets `dev`**, not `main`.

Do not merge the pull request unless you are asked to.
