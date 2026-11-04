#!/usr/bin/python3
"""Reject commit identity overrides and Codex attribution; never create a commit."""
import os
from pathlib import Path
import re
import subprocess
import sys


def git(*args, env=None):
    result = subprocess.run(['/usr/bin/git', *args], env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Unable to resolve configured Git identity.')
    return result.stdout.strip()


def configured_environment():
    # git -c settings propagate to hooks through GIT_CONFIG_PARAMETERS. Read
    # the persisted configuration without those ephemeral command overrides.
    removed = {'GIT_CONFIG_PARAMETERS', 'GIT_CONFIG_COUNT', 'GIT_CONFIG',
               'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_AUTHOR_DATE',
               'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL', 'GIT_COMMITTER_DATE'}
    return {key: value for key, value in os.environ.items()
            if key not in removed and not key.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_'))}


def identity(raw):
    match = re.fullmatch(r'(.*?) <([^<>]+)> \d+ [+-]\d{4}', raw)
    if not match:
        raise ValueError('Unable to parse resolved Git identity.')
    return match.group(1), match.group(2)


def main():
    try:
        env = configured_environment()
        expected = (git('config', '--get', 'user.name', env=env),
                    git('config', '--get', 'user.email', env=env))
        for variable in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
            if identity(git('var', variable)) != expected:
                raise ValueError('Commit blocked: Author and Committer must match persisted Git user.name/user.email. Remove identity overrides.')
        if len(sys.argv) > 2 and sys.argv[1] == 'commit-msg':
            message = Path(sys.argv[2]).read_text()
            for line in message.splitlines():
                if (re.match(r'^\s*(?:co-authored-by|signed-off-by|authored-by|generated-by|co-committed-by)\s*:', line, re.I)
                        and re.search(r'\bcodex\b|\bopenai\b', line, re.I)):
                    raise ValueError('Commit blocked: Codex/OpenAI bot attribution or co-authorship is not permitted.')
                if re.search(r'^\s*(?:🤖\s*)?generated (?:with|by)\s+.*\bcodex\b', line, re.I):
                    raise ValueError('Commit blocked: Codex generation attribution is not permitted.')
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
