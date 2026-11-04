# Git identity and restoring hooks on vast.ai

Both Author and Committer must use the identity configured in the checkout.
For this project it is currently `Minjae Bang <mb4882@columbia.edu>`. Use ordinary
`git commit`; do not override identity or copy it from older commits. Do not add
Codex/OpenAI bot attribution or a Codex co-author.

Git clones transfer tracked files, including `AGENTS.md` and the reviewed hook
sources in `scripts/git-hooks/`. They do not transfer local `.git/hooks/`, local
Git configuration, or this instance's `~/.codex/AGENTS.md`. Restore the following
settings and hooks on each new vast.ai clone. No training process needs to start
or restart for this setup.

## Configure the new clone

From the new project's checkout, configure repository-local identity:

```sh
git config --local user.name "Minjae Bang"
git config --local user.email "mb4882@columbia.edu"
git config --local user.useConfigOnly true
```

`user.useConfigOnly` prevents Git from guessing an identity if configuration is
missing. It does not prevent explicit command-level overrides; the instructions
and hooks address those. If you intentionally change your configured identity,
use the new identity for both fields rather than retaining the values above.

Before any commit, verify the configuration and both resolved identities:

```sh
git config user.name
git config user.email
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
```

Both identity commands must show `Minjae Bang <mb4882@columbia.edu>` followed by
Git's timestamp and timezone. Resolve a mismatch before proceeding. Do not use
identity environment variables, `--author`, command-scoped Git configuration, or
an attribution fallback to make a commit proceed.

## Review and restore the hooks

The vast.ai image needs `/usr/bin/git`, `/usr/bin/python3`, and the standard
`cmp` and `install` tools. Review the tracked guard, wrappers, and installer:

```sh
cat scripts/git-hooks/configured-identity-guard.py
cat scripts/git-hooks/prepare-commit-msg scripts/git-hooks/commit-msg
cat scripts/install-git-identity-hooks.sh
sh scripts/install-git-identity-hooks.sh
```

The installer uses the clone's common Git directory, so linked worktrees share
the hooks. It is safe to repeat when the installed sources match. It refuses to
overwrite a differing hook or a symlink, and refuses installation when any
`core.hooksPath` is configured. In those cases, preserve the existing policy and
review how to integrate these checks into it; do not disable it or reset the
hook path to bypass the refusal. Unrelated hooks are untouched.

The `prepare-commit-msg` hook checks Author and Committer against the persisted
configuration. The `commit-msg` hook repeats that check and rejects Codex/OpenAI
bot attribution. These local hooks support the ordinary Git commit workflow;
do not use alternate APIs or plumbing commands to bypass them.

## Verify without creating a test commit

Run the installed hooks directly, using a temporary message file:

```sh
sh -eu <<'SH'
hooks_dir="$(git rev-parse --path-format=absolute --git-common-dir)/hooks"
identity_message=$(mktemp)
trap 'rm -f "$identity_message"' EXIT
printf '%s\n' 'Verify configured Git identity' > "$identity_message"
"$hooks_dir/prepare-commit-msg" "$identity_message" message
"$hooks_dir/commit-msg" "$identity_message"
SH
```

Successful checks exit with status zero and create no commit. After a real,
requested commit, inspect its metadata with:

```sh
git log -1 --format='Author: %an <%ae>%nCommitter: %cn <%ce>'
```

Restoring these hooks does not change existing commits and requires no rebase,
amend, or force-push.
