### Batch sweep behavior

Current implementation submits the Cartesian product of:
`N × spacing × seed`.

Fixed per sweep: `duration`, `tx`, `ix`, `sparse_max_attempts` (override via CLI once per sweep).

Rationale
- Keeps the number of jobs predictable and avoids build/config explosion.
- You can still vary `tx/ix/duration` between sweeps if needed.

Future extension ideas
- Add optional ranges for `duration` or `tx` with careful limits.
- YAML configs mapping named experiment presets to sweep parameters.

