# Optional backends

## Model host and KernelGen

Offline compare, diagnose, hotspots, regions, roofline, context, and knowledge use
only the standard library. They do not start models/devices or depend on KG/KGS.
The skill guides interpretation; the host supplies the model. KG may consume context
JSON and use its Coder role to generate candidates. No KG internal patch is required;
other hosts can invoke the same interfaces.

## KGS evaluation

Adapters use the public HTTP protocol and preserve raw requests/results. Supply the
service address explicitly; there is no private-machine default.

```sh
hgkda kgs status --server http://127.0.0.1:19661
hgkda kgs upload --server http://127.0.0.1:19661 --archive operator.tar
hgkda kgs evaluate --server http://127.0.0.1:19661 --request request.json --review review.json --run-directory /path/out/new-run
```

Requests contain a candidate and the `binding`, `implementation`, and `settings`
accepted by the current KGS version. The pinned KGS client constructs contract bundles;
upload verifies SHA. Default to one request. `num_trials` repeats timing inside a
request, not independent high-score retries.

If contract and candidate both export `run(A)`, keep the contract name and adapt
candidate bindings explicitly. KGS rejects conflicting `run` and operator-name
functions. A Python preflight does not establish that KGS reconstruction can load
the interface; return failures to the generating model and evaluate the new source.

Screen known paths and review all code/fallback paths before submission. After a timeout,
query the original `operation_id` before deciding to retry; never replay automatically.
Static screening is not a full strict-Triton audit or formal performance acceptance.

## Hardware collection

Collection reads explicit JSON: `argv`, `source`, `source_sha256`, and optional `cwd`,
`environment`, `timeout_seconds`. `argv` is an argument array for `msprof op`;
`--run-directory` selects a new output directory. Success alone does not prove a complete capture.

Initialize CANN, device ownership/locks, and the target environment before collection.
Keep target work unchanged. Collection/evaluation use exclusive device access;
logs, environments, and caches live outside the repository.

## Measurement scopes

KGS full-call timing is the formal score. Hardware profiling and simulation have
separate costs. CANN/Ascend C and PyTorch references remain distinct; do not switch
reference scope to inflate candidate speedup.
