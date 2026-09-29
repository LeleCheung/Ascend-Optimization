# Read-only operator contract export

`POST /operator-contract` accepts the same `InspectRequest` binding as `/inspect`: an installed `catalog_name` or an internal `bundle_id`, plus the Definition name. `/status.capabilities.operator_contract.enabled` advertises support. No KG or KGS release number substitutes for this capability check.

The response contains the binding, evaluator kind, Catalog protocol version, full Definition (including reference source), and declared correctness/timing workloads. This lets a local KG prepare optimizer input from the target KGS Catalog, without assuming its local package data matches the remote installation. Python callers use `kernelgen_server.client.get_operator_contract()`.

This endpoint reads the Catalog without importing its oracle or FlagGems, running pytest, acquiring a device slot, or performing readiness checks. In particular, an adapter Catalog may declare no workloads: its executable case list and benchmark fingerprint still come from `/inspect`. Contract export does not replace `/inspect`, Preflight, Eval or source review. Exported code remains untrusted source text on the client and must not be imported to infer target capabilities.

Serialization preserves explicitly supplied null defaults while omitting unspecified defaults. Existing input-factory recipes are returned unchanged. Only named server-owned Catalogs and validated Bundle bindings are accepted, never arbitrary client filesystem paths. Missing content returns 404; invalid bindings or content return 422. The API exposes source to trusted clients and retains the existing loopback / SSH stdio proxy deployment boundary.

No Catalog is installed or modified, and no new version/tag is published by this change. This additive endpoint leaves the existing `/inspect` schema unchanged. Client runs must freeze the returned contract and separately validate the target execution identity; this read endpoint is not a transaction that locks the Catalog for the lifetime of an optimization run.
