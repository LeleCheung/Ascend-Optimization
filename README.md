# HG-KDA / Ascend-Optimization

Hardware-Grounded Kernel Design Agent (HG-KDA) connects paired hardware profiling,
white-box simulator analysis, source/IR inspection, and full operator evaluation.

| Directory | Purpose | Start here |
| --- | --- | --- |
| `hgkda/` | Installable diagnostics, runtime adapters, and agent skill | [Package guide](hgkda/README.md) |
| `operators/` | Delivered versions, contracts, checks, and bound evidence | [Delivery guide](operators/README.md) |
| `development/` | Software tests, engineering tools, research, and historical records | [Developer guide](development/README.md) |

```sh
python -m pip install ./hgkda
python -m hgkda doctor
python -m hgkda.runtime --help
```

The public package and selected deliveries work without `development/`.
Keep credentials, environments, caches, logs, and active runs outside the repository.

The main delivery index records **26 accepted operators out of 30**, plus nine
earlier historical operators. Later GQA results and source/IR provenance are indexed
separately in the [developer guide](development/README.md). Engineering cleanup does
not change scores, contracts, or candidate identities.

See [architecture](development/docs/architecture.md),
[language policy](development/docs/language-policy.md), and [maintenance rules](AGENTS.md).
Hardware-statistics spreadsheets are indexed under [development/data](development/data/README.md).
