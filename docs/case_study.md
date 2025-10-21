### Case study specifics

This repo includes components tailored to the RPL battery-aware study:
- Parser: `core/parse/rpl_log_to_csv_v3.py` expects client/server log tags and emits one row per mote.
- DB loader and schema under `core/db/` target a simple `runs`/`motes` layout.
- Firmware in `case_study_rpl/firmware/` and data processing helpers in `case_study_rpl/data_processing/` are provided as reference implementations.

Reuse guidance
- If your firmware logs differ, copy and adapt the parser or write your own log parser.
- If you need different run metadata, adjust the DB schema and loader scripts.

