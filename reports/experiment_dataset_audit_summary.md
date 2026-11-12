# IoT Sentinel AI — Consolidated Experiment Dataset Audit

Read-only audit. Raw experiment files were not modified.

- Primary runs passed: 9/9
- Primary CSV records: 2385
- Pilot runs passed: 1/1
- Pilot CSV records: 33

| Role | Run ID | Result | Configured rate | Measured rate | Submitted | ACKed | Received | Rejected | Issues |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| primary | `normal_baseline_v2_20261108T130132Z_0d8534df` | PASS | None | None | None | None | None | None | None |
| primary | `normal_baseline_v2_20261111T174556Z_d6e153b1` | PASS | None | None | None | None | None | None | None |
| primary | `normal_baseline_v2_20261111T175428Z_b01e38fe` | PASS | None | None | None | None | None | None | None |
| primary | `flood_v2_2rps_run01_20261111T161433Z_13332d0b` | PASS | 2.0 | 2.0 | 60 | 60 | 125 | 0 | None |
| primary | `flood_v2_2rps_run02_20261111T180926Z_39e51e4c` | PASS | 2.0 | 2.0 | 60 | 60 | 125 | 0 | None |
| primary | `flood_v2_5rps_run01_20261111T164008Z_05155788` | PASS | 5.0 | 5.0 | 150 | 150 | 215 | 0 | None |
| primary | `flood_v2_5rps_run02_20261111T181124Z_80434161` | PASS | 5.0 | 5.0 | 150 | 150 | 215 | 0 | None |
| primary | `flood_v2_10rps_run01_20261111T165033Z_8d2db887` | PASS | 10.0 | 10.0 | 300 | 300 | 365 | 0 | None |
| primary | `flood_v2_10rps_run02_20261111T181318Z_de9827ee` | PASS | 10.0 | 10.0 | 300 | 300 | 365 | 0 | None |
| pilot | `normal_baseline_v2_20261108T125829Z_f05f1992` | PASS | None | None | None | None | None | None | None |

## Interpretation

- Configured and measured submission rates are distinct measurements.
- Publisher acknowledgements and collector receipt counts are distinct measurements.
- Random seeds are reported only when present in metadata; missing seeds are not inferred.
- PCAP files are checked for presence and non-zero size; packet-level verification is separate.
- Passing this audit does not establish model performance or dataset sufficiency.
