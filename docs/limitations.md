### Limitations (scope and known constraints)

- Case-study parser/DB: `core/parse/rpl_log_to_csv_v3.py` and `core/db/*` are tailored to the RPL firmware in this repo. If your firmware logs differ, copy and modify them.
- Platform validation: Default is `z1`. `sky`/`wismote` specs are provided as previews and not validated end-to-end here.
- Radio model: UDGM with fixed `tx`/`ix` per run. No fading/interference dynamics beyond ranges; extend Cooja config if needed.
- Time compression: Energy currents are scaled uniformly (multiplier) to compress simulated time. Relative behaviors are preserved under uniform scaling, but absolute lifetimes depend on the scale.
- Sweep geometry: Batch sweep is a Cartesian product of `N × spacing × seed`. Other parameters are fixed per sweep. Extending ranges can quickly explode job count.
- Headless only: The pipeline assumes headless Cooja via Gradle; no GUI workflows are included here.

