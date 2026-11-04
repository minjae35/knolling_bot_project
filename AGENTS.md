# Git identity and commit attribution

For future commits, both Author and Committer must use the repository's configured
Git identity, currently `Minjae Bang <mb4882@columbia.edu>`.

- Read `git config user.name` and `git config user.email` from this checkout;
  verify both `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT` match.
- Use plain `git commit`. Do not override identity with `-c user.*`, `-c author.*`,
  `-c committer.*`, `--author`, identity environment variables, or command-scoped
  config. Do not copy identity from Git history or use a bot fallback.
- Do not add Codex/OpenAI bot attribution or a Codex co-author.
- Keep the local identity hooks enabled. Do not bypass them with alternate hook
  paths, `--no-verify`, plumbing commits, or external commit APIs.
- If identity is missing or inconsistent, resolve it before committing. Honor
  future changes to the user's configured identity rather than copying history.
- Restore missing hooks using [the Git identity guide](docs/git-identity.md).
