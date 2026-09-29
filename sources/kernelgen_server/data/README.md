# Catalog data

The top-level Catalog directories are usable by the current Server and use one
of its supported API versions (`v6.0` or `v6.2`):

- `flaggems-adapter-definitions`: Definition-only Catalog evaluated by the
  FlagGems framework adapter (`v6.0`);
- `flaggems-native`: native per-operator FlagGems Catalog (`v6.2`);
- `kernelgenbench`: migrated KernelGenBench Catalog (`v6.2`).

Unsupported historical Catalogs are retained under `.old/`. Do not pass an
archived Catalog to the current Server.

KGS wheels bundle every supported top-level Catalog so `builtin_catalog_path()` works after a non-editable installation. The archived `.old/` Catalogs are intentionally excluded.
