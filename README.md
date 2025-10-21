# Cooja-Automation-ML
Automation pipeline for large-scale Cooja simulations. Proof of concept for training ML on a dataset of simulations.

This directory contains a cleaned, publishable subset of the automation used to run headless Cooja simulations and parse results. It is intended to live alongside the original repository without overwriting legacy scripts.

Highlights:
- Modular CSC generation with a pluggable PlatformSpec (`z1` default; `sky`/`wismote` preview).
- Per-run, headless execution pattern (JSON → CSC → log → CSV).
- Parsing utilities including an XY/XYZ augmenter for coordinates extracted from `.csc`.

Examples

1) Generate a CSC from a topology JSON:

```bash
python3 core/topology/topology_generator.py grid -n 5 -s 10 -o sample_topologies/grid5.json
python3 core/csc/csc_generator.py sample_topologies/grid5.json artifacts/example/sim.csc --build-root artifacts/example/build
```

2) Select a different platform (preview):

```bash
python3 core/csc/csc_generator.py sample_topologies/grid5.json artifacts/example_sky/sim.csc --platform sky
```

3) Augment a parsed CSV with coordinates (optional Z support):

```bash
python3 core/parse/rpl_log_add_xyz_to_csv.py --grid artifacts/example/sim.csc --input artifacts/example/results.csv --output artifacts/example/results_xy.csv
# With Z column
python3 core/parse/rpl_log_add_xyz_to_csv.py --grid artifacts/example/sim.csc --input artifacts/example/results.csv --output artifacts/example/results_xyz.csv --enable-z --z-col-name z
```

Notes
- Batch sweep currently sweeps the Cartesian product of N × spacing × seed. Other parameters (duration, tx, ix) are fixed per sweep; extend as needed.
- Parser and DB loader are tailored to the case study; they can be adapted for other firmware/log schemas.
- Platform specs beyond `z1` are previews; validate before large sweeps / simulation runs.

### Testing

- Quick: `make test` (runs `pytest -q` against `tests/`)
- Direct: `pytest -q cooja-automation-repo/tests`

Notes:
- Tests are self-contained and do not require Cooja; they validate topology JSON/CSC generation, parser behavior on a tiny synthetic log, and XY/XYZ augmentation.

### Prerequisites for simulation (optional)

To actually run headless Cooja simulations via the run scripts, you need a working Contiki-NG+Cooja setup and Java/Gradle:
- Install Contiki‑NG and Cooja per the official docs: [Contiki‑NG documentation](https://github.com/contiki-ng/contiki-ng)
- Ensure `--cooja-root` points to your Cooja folder (typically `<contiki-ng>/tools/cooja`).
- A JDK 17+ is required; you can pass `--java-home` to the run scripts if needed.

If you only want to use the topology/CSC tools and parsing utilities, you do not need Contiki‑NG/Cooja.

