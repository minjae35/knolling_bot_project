# Autonomous experiment readiness — BLOCKED

Verified on 2026-11-08 UTC. No autonomous optimization loop or GPU training was started. No two-cycle experiment verification has passed. This is a prerequisite failure report, not an implementation-completion report.

## Preserved baseline

- Primary checkout: `/workspace/knolling_bot_project`, clean `main` at `28bab156ded655857713674e97fc7bf572f76262`.
- Isolated checkout: `/root/.codex/worktrees/knolling-autonomous/knolling_bot_project`, branch `experiment/autonomous-knolling`.
- Registry SHA-256: `7d28df2bbda50c7a676284f7f606b52fd8fa85cbb487b5187f498956e4425bd6`.
- Arrangement Best SHA-256: `d92ad1ffcabf1842c7d805e8a8445b3dfca7cf3bfad05aa92318db3c010ba3d9`.
- Perception Best SHA-256: `c02dc5e9501a39cfcb9d1de8d512ace765050d675155e66bbde0ec2889105491`.
- Best checkpoint hashes match the existing registry. Dataset, checkpoint and evaluation files were not modified. Model evaluations were not rerun.
- Final free space observed: 51,825,811,456 bytes (48.27 GiB). No GPU compute process was reported by `nvidia-smi`.

## Actual capability checks

Installed CLI: `codex-cli 0.166.0`. Existing configured model and authentication were retained; no credentials were printed or changed.

1. `codex login status`: `Logged in using ChatGPT`.
2. Noninteractive model-only request: returned `AGENT_AUTH_OK`; thread `01a1bb0c-975e-7822-bcbe-79e1e97da1e7`.
3. Workspace-write file probe: `apply_patch` failed to create `agent-capability-probe.txt`; thread `01a1bb0d-08ab-7d42-857b-b5d54ef39dd2`.
4. Workspace-write probe with the exact worktree explicitly included: even `pwd` failed; thread `01a1bb0d-61b7-7022-a6b8-62e762ef085f`:

```text
bwrap: No permissions to create a new namespace, likely because the kernel does not allow non-privileged user namespaces. On e.g. debian this can be enabled with 'sysctl kernel.unprivileged_userns_clone=1'.
```

5. The installed CLI's official legacy Landlock fallback was checked once, retaining `workspace-write`. It also failed; thread `01a1bb0e-453f-72a1-a211-8559b96729af`:

```text
thread 'main' (152520) panicked at linux-sandbox/src/linux_run_main.rs:385:9:
filesystem-restricted execution requires bubblewrap to isolate app-server sockets
```

No probe file was created. These are capability checks, **not scientific experiment cycles**. Verified experiment cycles: **0**. GPU training time: **0**. No persistent tmux worker was launched, so there is no active-loop attach/resume command.

`unshare --mount --user --map-root-user true` independently returned `Operation not permitted`. The visible `kernel.unprivileged_userns_clone` setting was already `1`, and `user.max_user_namespaces` was `2062206`. Therefore simply setting that sysctl to 1 is not a verified remedy. The precise host/container restriction remains unconfirmed.

## Required recovery before execution

Use a host/container where the CLI can run its protected workspace-write tools, or have the instance administrator diagnose and correct namespace creation restrictions. Preserve the filesystem isolation protecting the original assets. No unrestricted-root substitute, CLI security bypass, kernel change, or authentication change was applied.

After the environment is repaired, first verify an actual noninteractive edit/read/test in the isolated worktree. Only then implement and validate the controller, immutable evaluation contract, source-family validation/test separation, persistent budgets, sequential GPU locking, recovery, and two consecutive adaptive experiment cycles. Start the overnight loop only after those checks pass. Retain the user-approved 10-hour wall / 5-hour cumulative GPU budgets and 12-GiB free-space floor; no budgeted training has begun here.

## References checked

- Official noninteractive CLI documentation: https://developers.openai.com/codex/noninteractive/
- Official sandbox implementation and explicit Landlock restriction: https://github.com/openai/codex/blob/main/codex-rs/linux-sandbox/src/linux_run_main.rs

The current repository also has a future family-pilot contract in `src/knolling_model/recovery/family_manifest.py`, but no matching materialized manifest was found in the initial filename scan. Independent Arrangement/E2E evaluation data readiness is not yet established. No final-success claim can be made.

## Focused sandbox diagnosis — 2026-11-08 12:12 UTC

Scope: diagnosis only; no training, model/source changes, package installations, global CLI configuration changes, host policy changes, or elevation. Python evidence collection used the `knolling` Conda environment.

Raw command arguments, exit codes, stdout and stderr are recorded in:
`artifacts/sandbox_diagnostics/20261108T121231Z/summary.json` (relative to this isolated worktree). Individual command JSON files are beside it.

| Check | Actual result |
| --- | --- |
| Ordinary `/bin/pwd`, outside the child CLI sandbox | Exit 0 |
| `codex sandbox -P :workspace -C <worktree> /bin/pwd` | Exit 1, namespace creation denied |
| Same sandbox creating a tiny `probe.py`, then reading it | Exit 1 before shell execution; probe was not created |
| Same sandbox from a neutral newly created `/tmp` directory | Exit 1, same namespace error |
| `strace ... unshare --user /bin/true` | `unshare(CLONE_NEWUSER) = -1 EPERM` |
| Mount namespace probe | `unshare(CLONE_NEWNS) = -1 EPERM` |
| User namespace probe after dropping to UID/GID 65534 | Exit 1, `Operation not permitted` |
| Explicit legacy Landlock with `:workspace` | Exit 101, explicit unsupported-policy guard |

The minimal code-write check attempted only `assert 1 + 1 == 2` in a new diagnostic file. Its failure is not a successful edit test. No scientific experiment was run.

### Confirmed boundary and remaining uncertainty

- CLI version remains 0.166.0. Its bundled `bwrap` exists and executes its version command. It is mode 775, not setuid. Missing system `bwrap` is therefore not the immediate failure.
- Container reports Ubuntu 24.04.5 and host kernel 6.8.0-142-generic.
- AppArmor label is `docker-default (enforce)`; seccomp is active (`Seccomp: 2`, one filter).
- `kernel.unprivileged_userns_clone=1`; `user.max_user_namespaces=2062206`; `kernel.apparmor_restrict_unprivileged_userns=1`.
- Effective capabilities lack `CAP_SYS_ADMIN` and `CAP_MAC_ADMIN`. This is observed, **not** a claim that granting these broadly is required or acceptable.
- No Docker/Podman management socket, usable AppArmor policy listing, `apparmor_parser`, or local `bwrap-userns-restrict` profile was found in the checked locations.
- The confirmed failure is namespace creation denied by the current execution environment, before project code runs. It is not specific to the worktree location, Git permissions, model authentication, or GPU configuration. The evidence does **not** distinguish a seccomp denial from an AppArmor denial; host audit/runtime policy inspection is needed for that attribution.

### Supported safe route on this instance

Official documentation recommends a targeted `bwrap` AppArmor profile on affected Ubuntu hosts, rather than globally disabling the user-namespace restriction. That is a conditional host-administrator remedy, **not a verified fix for this Docker container**. The inherited container seccomp/AppArmor rules must also permit the required operations. Installing a package inside this container alone has not been demonstrated to solve those restrictions.

Current official permission documentation explicitly rejects legacy Landlock for filesystem-restricted policies because it cannot isolate app-server Unix sockets. The earlier description of it as an available fallback should therefore not be interpreted as a supported working alternative for this CLI version.

No application-level configuration fix retaining the requested isolation has passed here. Do not change to unrestricted execution, privileged containers, unconfined policies, setuid changes, or globally disabled AppArmor/seccomp to make the probe pass. A host administrator should diagnose the relevant denial and apply only a narrowly scoped supported policy, then repeat both positive workspace-edit and negative outside-workspace-write tests.

### Recommended alternative: separate controller and GPU worker

This is a proposed deployment, not an already verified remote system:

1. Run Codex CLI and the persistent orchestrator on an always-on Linux VM whose normal workspace sandbox passes, or on an awake macOS machine using the supported Seatbelt sandbox. Keep an independent experiment checkout there. Before deployment, verify noninteractive `pwd`, edit/read/test, denial of writes outside the workspace, and two sequential adaptive cycles on that actual host.
2. Keep Vast.ai as a GPU worker. Transfer versioned experiment code and validated configuration into a new per-run directory; keep the current main checkout, datasets and Best checkpoints protected. Do not move the Agent back into the same incompatible container through Remote SSH.
3. Use a constrained job interface for submit/status/result retrieval, not arbitrary Agent-controlled root SSH. The worker must enforce allowed paths, read-only protected assets, sequential GPU locking, disk limits and persisted wall/GPU budgets. An SSH forced-command endpoint or equivalent authenticated worker service can implement that interface; neither has been deployed here.
4. The controller analyzes returned metrics/logs and generates the next code revision. GPU jobs stay in tmux on Vast.ai. Durable controller state survives reconnection, and duplicate job IDs cannot start duplicate training. Only new result directories receive writes.

The different host's sandbox, authentication and worker connectivity remain untested. Merely running the same CLI through a remote editor on Vast.ai does not change the failing namespace boundary.

### Preservation check

After diagnosis, primary `main` still equals `28bab156ded655857713674e97fc7bf572f76262` and has no working-tree changes. The registry and both Best checkpoint SHA-256 values still match the values above. No original Dataset, Checkpoint or result file was written. Only this worktree's diagnostic outputs and report were added/updated. No commit or push was performed for this diagnosis.

Official references retrieved for this follow-up:

- https://learn.chatgpt.com/docs/sandboxing — platform prerequisites, bundled helper, targeted Ubuntu AppArmor profile, macOS Seatbelt.
- https://learn.chatgpt.com/docs/permissions — filesystem-restricted Linux policies require Bubblewrap; legacy Landlock is rejected.
- https://learn.chatgpt.com/docs/non-interactive-mode — noninteractive CLI execution interface.
