# Validation performed on 2026-10-08

- Four benchmark producer files match the hashes recorded in v8.
- All 61 archived code/protocol files pass the release SHA-256 check.
- The numerical runner verified 29 separately supplied input files, then ran the
  original v8 table scripts in a new local directory.
- All 20 generated CSV reference tables match their saved v8 counterparts;
  their SHA-256 values also match exactly. This includes the 29-representation
  primary table, clustering-grid summaries, context summaries, and the corrected
  time-course display tests and screening summaries.
- The main test suite reports **17 passed**. One test invokes the eight original
  time-course framework tests; another invokes the original CPU distance and
  transformation oracles. Existing synthetic CPU training, checkpoint and Align
  tests also passed.
- The wheel was built through `setuptools.build_meta` (the separate `build`
  frontend is not installed in the local validation environment).

The numerical comparison hashes are recorded in [VALIDATION.json](VALIDATION.json).
Input tables and locally generated per-cell tables are excluded from Git.

Local source-tree execution used Python 3.13 with NumPy 2.2.6, pandas 2.3.2 and
SciPy 1.16.2; this is not a clean installation test of the package's existing
NumPy < 2 dependency constraint. The GitHub CI configuration uses Python 3.11
and dependency resolution from `pyproject.toml`. CI completion must be checked
on GitHub separately.

No full-cohort GPU inference, model pretraining, new biological experiment or
multiomic checkpoint recovery was performed for this release. The checks do
not establish those outcomes or confirm independent biological replication.
