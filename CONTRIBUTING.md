# Contributing

This document describes the workflow and rules that all contributors must follow.

## Workflow Overview

Every change follows this path:

```
Issue → Branch → PR (linked to issue) → CI passes → Merge → Branch auto-deleted
```

## Step 1: Create an Issue

Every change starts with a GitHub issue. Use one of the provided templates:

- **Bug Report** — for bugs and unexpected behavior
- **Feature Request** — for new features and improvements
- **Documentation** — for docs improvements or missing content

## Step 2: Create a Feature Branch

Branch from `main` using one of these naming conventions:

| Prefix | Use for | Example |
| -------- | --------- | --------- |
| `feature/` | New features | `feature/42-add-rate-limiting` |
| `fix/` | Bugfixes | `fix/17-correct-threshold-calc` |
| `docs/` | Documentation | `docs/8-update-setup-guide` |

Format: `<prefix>/<issue-number>-short-description`

```bash
git checkout main
git pull origin main
git checkout -b feature/42-add-rate-limiting
```

## Step 3: Make Changes and Commit

- Write small, focused commits
- Use conventional commit messages:
  - `feat: add rate limiting configuration`
  - `fix: correct threshold calculation`
  - `docs: update setup guide`

## Step 4: Open a Pull Request

1. Push your branch and open a PR against `main`
2. **Link the issue** — use `Closes #42` in the PR description, or link from the sidebar
3. Fill out the PR template (it loads automatically)

## Step 5: Review and Merge

- All CI checks must pass before merge
- Squash merge is preferred
- The branch is automatically deleted after merge (`delete_branch_on_merge` is enabled)
