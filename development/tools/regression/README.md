# Fixed-candidate independent-entry regression

Verify existing candidates through the independent HG-KDA interface without old KG
model scheduling/controllers. KGS runs full device evaluation, supplementary checks
use frozen checkers, and failures preserve historical acceptance.

## Prepare

Run in a Linux environment with the current package installed:

```sh
python development/tools/regression/prepare.py --repository /path/Ascend-Optimization --archive-project /path/archive/project/Ascend-Optimization-v1 --original-project /original/Ascend-Optimization/project/Ascend-Optimization-v1 --run-directory /path/runs/new-regression
```

This historical batch selects 25 fixed accepted entries from the frozen experiment
index. Verify candidate/review hashes; copy contracts/checkers with original bytes,
including CRLF; verify formal request bundle identity. Where old KG snapshots retain
only solution/identity/evaluation-contract, reconstruct the same HTTP request and
record that recovery explicitly.

Enumerate verified absolute-path native-library/host-bridge dependencies. Execution
may create temporary compatibility links and remove them after normal completion;
strict device work remains Triton.

## Execute

Use an independent session such as tmux:

```sh
python development/tools/regression/run.py --plan /path/runs/new-regression/plan.json --container shared-container --device 7 --port 19663 --lock /path/runs/physical-card-7-campaign.lock --device-python /path/python3.11 --cann-environment /path/cann/set_env.sh
```

The controller locks the allocated device, starts a dedicated KGS service, screens
source, uploads, evaluates once per operator, and runs both frozen semantic checks.
Preserve timing settings, source, contracts, requests, operation IDs, results, and status.
Skip terminal jobs on recovery. For an existing submission without results, query the
original operation_id; never replay blindly. Preserve unresolved service/dependencies.
Do not append high-score retries or restart shared containers. Stop only the identified
batch service.

## Summarize and audit

```sh
python development/tools/regression/summarize.py --run-directory /path/runs/new-regression --output /path/runs/new-regression/summary-final.json
python development/tools/regression/audit.py --run-directory /path/runs/new-regression --output /path/runs/new-regression/audit-final.json
```

Outputs include English tables, per-case CSV, and JSON. Acceptance requires full
correctness, strict review, both original semantic checks, and >0.8x a qualified CANN
reference. Process exit is insufficient. Compare old/new candidate and reference times
separately; aggregate score ratios alone are not candidate gains.

Audit source/review/checker SHA, original bundle/request, timing settings, complete
case sets, check counts, and reconstructed per-case ratios/geometric means.
`evidence_verified` and `accepted` are distinct; retain below-threshold complete evidence.

Archived batches support offline Windows/Linux auditing:

```sh
python development/tools/regression/audit.py --run-directory development/experiments/regressions/20261109/fixed-25 --original-run-directory /data/hanle/ascend-optimization/runs/independent-regression-20261109-v3 --verify-transfer --output ../hgkda-output/new-archive-audit.json
```

Output must not exist; create its parent. `--original-run-directory` maps only in
memory; `--verify-transfer` verifies all lengths/SHA without editing originals.
Use `git lfs pull` before auditing actual archives/binaries.

## Targeted support

[Operator checks](../operators/README.md) and native-reference qualification remain
separate. Old direct candidate builders are [retired](../archive/README.md), with
[historical results](../../reports/regression-20261109.md) intact. CC/DeepSeek owns changes.
`capture_compiled.py` inspects actual Triton artifacts and rejects non-metadata ATen
calls in candidate scope; combine it with full checks/review. `compare_previous.py`
compares actual same-source per-case candidate/reference times without score picking.
