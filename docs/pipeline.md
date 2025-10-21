### Pipeline overview

Stages and persisted artifacts per run:
- Topology JSON → `core/topology/topology_generator.py`
- CSC file → `core/csc/csc_generator.py`
- Headless Cooja log (via Gradle) → `cooja.log`
- Parsed CSV → `core/parse/rpl_log_to_csv_v3.py`
- XY/XYZ augmentation → `core/parse/rpl_log_add_xyz_to_csv.py`
- Metadata → `run_meta.json`, `roles.json`

Default output location is `artifacts/<run_id>/` with per‑run, self‑contained files.

Notes
- The parser and DB loader are tailored to the RPL case study; adapt them if your firmware logs differ.
- Graphical Cooja is not required; all commands are headless.

