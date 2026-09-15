Real run data is written here and ignored by Git. No fabricated model scores are shipped.

Each run has `results.csv`, `summary.csv`, and `metadata.json` in a timestamped directory.
`evaluations.sqlite` stores runs and per-case records; `latest_results.csv` points to the
last fully attempted run. An interrupted run never replaces latest or baseline.
Use `python -m regression.baseline <run>/results.csv` to establish a baseline.
