# Windows control host and 910B evaluation

Windows stores the control checkout and issues remote commands. Configured 910B
servers/containers provide CANN, Triton-Ascend, torch_npu, and KGS. Missing local
Windows torch_npu is separate from remote availability.

## Verify identity

```sh
ssh <910b-alias> hostname
ssh <910b-alias> git -C <remote-repository> status --short --branch
```

SSH may use ssh-agent or an explicit IdentityFile. `ssh-add -l` only tests agent access;
use read-only remote commands for connection checks. Verify repository, source SHA,
container, interpreter, device mapping, and KGS `/status`; branch names alone are insufficient.
Never commit credentials or personal SSH configuration.

Historical environment (2026-11-16): `/data/hanle/ascend-optimization/Ascend-Optimization`,
container `tle_yy`, physical card 7 exposed as logical `npu:0` by
`ASCEND_RT_VISIBLE_DEVICES=7`; 910B4, CANN 9.0.0, Triton 3.5.1, FlagTree 0.6.0.
Treat this as a drift-diagnosis snapshot; actual configuration comes from the target
interpreter and service status. Verify SDK paths inside the executing container.
KGS and docker-exec supplementary checks use the same container/device mapping.

## Evaluate

1. Verify allocation, `npu-smi info`, and service status. Idle utilization does not
   establish exclusive ownership. Serialize tests with one worker/project lock.
2. If needed, start only an identified project service on a free independent port,
   with separate run/profile/cache directories and `--max-workers 1`. Verify protocol,
   device, software, and scheduler through `/status`.
3. Prepare a new external run using `python -m hgkda.runtime prepare` with explicit
   `--operator-directory`, `--package`, `--environment`, and `--run-directory`.
4. Review `source-reviews/<SHA>.json`, then run `python -m hgkda.runtime evaluate
   --config <run>/config.json --port <kgs-port>` once. Bound PyTorch/CANN checks and
   full KGS results remain separate. Inspect audit-final.json, evaluation/result.json,
   and each check; process exit alone is insufficient.
5. Confirm workers are idle and stop only this run's verified service PID/identity.
   Avoid bulk process-name termination, VCD service changes, or shared-container restart.

Keep source SHA, request/raw results, checks, environment, and lifecycle records in a
new external directory. Do not update release ZIPs or overwrite scores; compare
measurements side by side.

Historical sample validation on 2026-11-16 covered scatter_add, native_group_norm,
and cross_out with full KGS and both checks. Remote records remain under
`/data/hanle/ascend-optimization/runs/codex-sample-retest-20261116/`; local external
copy: `deliveries/hgkda-sample-retest-20261116/`. The 26-operator ZIP was unchanged.
