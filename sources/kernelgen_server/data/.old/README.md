# Unsupported historical Catalogs

This directory archives Catalogs whose `api_version` is not supported by the
current KernelGen Server. They are retained only for provenance, migration and
training-data inspection:

- `flaggems-v5-to-test`;
- `akg-bench-lite-v5`;
- `flaggems-v5`;
- `pytorch-v5`.

The current Server accepts only `v6.0` and `v6.2`; loading any Catalog in this
directory must fail with `unsupported catalog API version`. New runnable data
must be migrated to a supported schema and placed at the top level of `data/`.
