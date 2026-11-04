#!/bin/sh
# Restore the reviewed identity hooks without replacing unrelated hook policy.
set -eu

repo_dir=$(git rev-parse --show-toplevel)
common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
hooks_dir="$common_dir/hooks"
source_dir="$repo_dir/scripts/git-hooks"

if git config --get core.hooksPath >/dev/null; then
    echo 'A custom core.hooksPath is configured. Review and integrate the identity checks there; no hooks were changed.' >&2
    exit 1
fi

for name in configured-identity-guard.py prepare-commit-msg commit-msg; do
    if [ -L "$hooks_dir/$name" ] || { [ -e "$hooks_dir/$name" ] && ! cmp -s "$source_dir/$name" "$hooks_dir/$name"; }; then
        echo "Existing $name differs from the reviewed source. Review and integrate it; no hooks were changed." >&2
        exit 1
    fi
done

# Check both identities against persisted Git configuration before installing.
/usr/bin/python3 "$source_dir/configured-identity-guard.py" prepare-commit-msg
mkdir -p "$hooks_dir"
for name in configured-identity-guard.py prepare-commit-msg commit-msg; do
    install -m 755 "$source_dir/$name" "$hooks_dir/$name"
done
echo 'Git identity hooks installed for this clone and its linked worktrees.'
