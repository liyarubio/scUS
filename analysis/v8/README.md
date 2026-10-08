# v8 analysis code and reproduction scope

This release links the existing scUS model package to the analyses described in
`iclr2027_paper_revision_v8_20260926`. It contains code and protocol metadata.
Data tables, per-cell records, checkpoints, figures, the manuscript and patent
materials are not included. No public download of those inputs is asserted.

[Chinese evidence notes](EVIDENCE_CN.md) explain what the results support and
where the evidence stops. The earlier September 17 result snapshot remains a
historical release and must not be combined with these protocols.

## What can be run

The portable entry point recomputes numerical summaries from a separately
supplied, checksum-verified v8 manuscript input bundle:

```bash
python -m pip install -e '.[analysis,dev]'
python analysis/v8/reproduce.py --help
python analysis/v8/reproduce.py \
  --input-dir /path/to/iclr2027_paper_revision_v8_20260926/manuscript \
  --check-only
python analysis/v8/reproduce.py \
  --input-dir /path/to/iclr2027_paper_revision_v8_20260926/manuscript \
  --output-dir outputs/v8_recomputed
```

The output directory must be absent or empty and separate from the input tree.
The command verifies the input hashes, copies the required inputs into an
isolated work directory, executes the two unmodified v8 table scripts, and
compares generated tables with supplied saved reference outputs. It writes
`validation.json` and execution logs. Missing reference outputs are explicitly
reported; they are never counted as a successful comparison. Missing or modified
required inputs stop execution before output creation.

This command **does not run model training, embedding inference, clustering,
full gene-screen fitting, or the multiomic exporter**. It re-summarizes existing
benchmark/context statistics and recomputes the specified time-course display
rank tests. The input contract is in
[`protocols/statistics_inputs.json`](protocols/statistics_inputs.json).

## Code map

| Code | Role | Reproduction status |
|---|---|---|
| [`src/scus`](../../src/scus) | Model, pretraining, frozen U/S distances, shared-residual Align | Existing installable package; actual path is repository `src/scus` |
| `archive/manuscript/revision_tables.py` | 29-representation tables, clustering-grid and matched-context summaries | Exact v8 script; run by the portable entry point |
| `archive/manuscript/timecourse_tables.py` | Corrected time-course summaries and display rank tests | Exact v8 script; run by the portable entry point |
| `archive/manuscript/compose_figures.py`, `raster_helpers.py` | Six-figure composition from saved values, coordinates and selected pixels | Exact v8 scripts; require the complete local figure-input tree; not run by `reproduce.py` |
| `archive/benchmark/code/` | Preprocessing, full-input encoders, 29 readouts, distances, clustering and plots | Historical producer snapshot; external assets/environments and path configuration required |
| `archive/context/` | Matched-bin examples and historical context analysis | Historical snapshot; the exact earlier cache-export revision remains unresolved |
| `archive/diagnostics/` | Training-log, checkpoint and pairing diagnostics | Historical scripts also contain conditions outside the v8 display; filter protocols explicitly |
| `archive/timecourse/` | Expression-equivalence and minimum-distance-change screen | Historical exploratory screen, separate from the display rank tests |
| `archive/multiomics/` | Target-adapted RNA/ATAC analysis and ordered heatmaps | Historical source; exact distance-export checkpoint binding remains unresolved |

[`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) records original source locations
and SHA-256 for every copied file. The four producer hashes explicitly recorded
by v8 (`features.py`, `model_worker.py`, `transforms.py`, `velocity_worker.py`)
match the archived benchmark snapshot byte for byte. The snapshot was retained
in the v6 source package; these four hashes, not its directory name, establish
its association with v8. Other historical files are provenance references, not
proof of an exact historical export revision.

The vendored scUS code is isolated under `archive/benchmark/code/vendor` and is
not installed over `src/scus`. `JointModel.hidden()` is used by the historical
benchmark to return raw, pre-adapter hidden states under inference mode; its
class name does not imply joint fine-tuning in the frozen benchmark.

## Protocol boundaries that affect the result

- Pretraining selects up to 1,000 genes. The v8 clustering benchmark uses
  **untruncated full input**, with cohort-fitted normalization/moments and
  gene-coverage filtering. Running the default capped inference configuration
  is not an exact reproduction of the benchmark.
- The pooled benchmark comparator concatenates the mean U and mean S vectors
  over eligible paired genes. It is not the all-token pooled vector produced
  by the default `pooled_embedding.npy` convenience output.
- Gene-distance profiles use coordinate standardization and RMS distances over
  jointly observed entries. Missing pairs remain missing. The availability-mask
  baseline uses Jaccard distance. Other readouts have their own definitions in
  [`protocols/readout_protocols.json`](protocols/readout_protocols.json).
- Clustering uses 15 neighbors; the displayed setting is Leiden resolution 1,
  seed 42. The archived sensitivity grid has three resolutions and three seeds.
  Preprocessing/clustering use the evaluated cohort, so this is transductive,
  descriptive evaluation, not independent held-out generalization.
- The time-course display keeps the complete historical 12-test SKAP2 family
  and the separate six-test RPL23 family. Neither is the genome-wide discovery
  family. The stricter separate-U/S-equivalence screen has no passing records.
- Distance has no direction or time unit. It is not RNA velocity, a kinetic
  rate, causal regulation or a clinical score. A smaller distance is not itself
  evidence of better biology or better model performance.

## Historical producer requirements

The archive preserves original source bytes, including historical absolute
paths. Inspect and configure them before executing a producer in a new
workspace. `protocols/benchmark.example.json` retains numeric settings with path
placeholders; it is a reference template, not a complete runnable release of
all original dataset configuration. Required external assets include U/S input
matrices, vocabularies, orthology maps, model checkpoints and method-specific
software environments (for example scGPT, scFoundation/omicverse, scVI and
scVelo). Third-party implementation copies are not included.

Some historical multiomic scripts also import modules from the original
training workspace. This release publishes their source for inspection, but
does not claim a portable multiomic rerun or a recovered export checkpoint.
No GPU queue or historical producer is started by installation or CI.

## Checks

```bash
pytest
```

With the `analysis` extra installed, tests run the archived CPU distance and
preprocessing oracles in temporary directories and the original time-course
statistics tests. Existing model/training/Align tests remain in place. These
checks validate software behavior; they do not rerun the full biological
experiments. See [VALIDATION.md](VALIDATION.md) for the actual release checks.
