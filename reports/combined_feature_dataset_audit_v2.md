# IoT Sentinel AI — Combined V2 Feature Dataset Audit

- Overall result: **PASS**
- Files passed: **9/9**
- Combined rows: **126/126 expected**
- Combined columns: **36**
- Exact duplicate rows: **0**
- Duplicate predictor vectors: **0**
- Blank cells: **0**
- Infinite predictor cells: **0**

## Per-file results

| File | Rows | Columns | Result | Missing | Infinite | Exact duplicates | Predictor duplicates |
|---|---:|---:|---|---:|---:|---:|---:|
| `normal_original_v2.csv` | 30 | 36 | PASS | 0 | 0 | 0 | 0 |
| `normal_seed101_v2.csv` | 30 | 36 | PASS | 0 | 0 | 0 | 0 |
| `normal_seed202_v2.csv` | 30 | 36 | PASS | 0 | 0 | 0 | 0 |
| `flood_2rps_run01_v2.csv` | 6 | 36 | PASS | 0 | 0 | 0 | 0 |
| `flood_2rps_run02_v2.csv` | 6 | 36 | PASS | 0 | 0 | 0 | 0 |
| `flood_5rps_run01_v2.csv` | 6 | 36 | PASS | 0 | 0 | 0 | 0 |
| `flood_5rps_run02_v2.csv` | 6 | 36 | PASS | 0 | 0 | 0 | 0 |
| `test_flood_v2.csv` | 6 | 36 | PASS | 0 | 0 | 0 | 0 |
| `flood_10rps_run02_v2.csv` | 6 | 36 | PASS | 0 | 0 | 0 | 0 |

## Combined distributions

- Targets: `{"mixed": 24, "normal": 102}`
- Phases: `{"attack": 12, "mixed_boundary": 12, "normal_baseline": 90, "post_attack": 6, "pre_attack": 6}`
- Scenarios: `{"message_flooding": 36, "normal_baseline": 90}`

## Predictor statistics

| Feature | Count | Min | Max | Mean | Population SD |
|---|---:|---:|---:|---:|---:|
| `packet_count` | 126 | 10.0 | 111.0 | 19.19047619047619 | 21.17864838445292 |
| `mqtt_publish_count` | 126 | 10.0 | 111.0 | 18.928571428571427 | 21.13906282984804 |
| `total_bytes` | 126 | 3384.0 | 39332.0 | 6568.507936507936 | 7522.924105107459 |
| `mean_packet_size_bytes` | 126 | 316.2307692307692 | 354.34234234234236 | 336.966997529433 | 9.451509471572464 |
| `std_packet_size_bytes` | 126 | 0.8331955809010618 | 78.1521047655784 | 18.185289492485218 | 28.378886181303184 |
| `min_packet_size_bytes` | 126 | 46.0 | 340.0 | 265.96031746031747 | 120.48113866702288 |
| `max_packet_size_bytes` | 126 | 340.0 | 356.0 | 343.8412698412698 | 5.5953675545081785 |
| `packet_rate_per_second` | 126 | 1.0 | 11.1 | 1.9190476190476191 | 2.117864838445292 |
| `publish_rate_per_second` | 126 | 1.0 | 11.1 | 1.8928571428571428 | 2.113906282984804 |
| `mean_interarrival_seconds` | 126 | 0.08994615294716575 | 0.9038890838623047 | 0.7218022880641097 | 0.24784439648704576 |
| `std_interarrival_seconds` | 126 | 0.02565030670630354 | 0.8812846383080516 | 0.6980054606645621 | 0.23597772947937634 |
| `min_interarrival_seconds` | 126 | 1.1205673217773438e-05 | 0.001668691635131836 | 0.0008148938890487428 | 0.00036325692608578863 |
| `max_interarrival_seconds` | 126 | 0.10996890068054199 | 2.022458553314209 | 1.8519726518600705 | 0.513994663131681 |
| `unique_tcp_streams` | 126 | 1.0 | 2.0 | 1.1904761904761905 | 0.392676726249301 |
| `telemetry_message_count` | 126 | 10.0 | 111.0 | 18.928571428571427 | 21.13906282984804 |
| `mean_temperature` | 126 | 23.113999999999997 | 29.770909090909093 | 26.432944125108033 | 1.5595685195464506 |
| `std_temperature` | 126 | 0.6212929733046124 | 7.136321134789478 | 2.7343320180511923 | 1.6581287484952774 |
| `mean_humidity` | 126 | 43.65416666666667 | 59.33090909090909 | 50.70552312933015 | 3.2282247067323326 |
| `std_humidity` | 126 | 1.4661495369623816 | 17.391892487752894 | 7.148946570792047 | 3.977604024196948 |
| `mean_battery` | 126 | 86.02272727272727 | 91.201 | 89.03187282366565 | 1.5102353892415938 |
| `std_battery` | 126 | 1.1308311440589485 | 3.978971585225641 | 2.7989647026956037 | 0.7582734425018517 |
| `unique_devices` | 126 | 3.0 | 4.0 | 3.1904761904761907 | 0.392676726249301 |

## Selected feature correlations

| Feature pair | Pearson correlation |
|---|---:|
| `packet_count / mqtt_publish_count` | 0.9997858554600778 |
| `packet_rate_per_second / publish_rate_per_second` | 0.999785855460078 |
| `unique_tcp_streams / unique_devices` | 1.0000000000000002 |
| `packet_count / total_bytes` | 0.9998431326931162 |

## Errors

- None reported.

## Warnings

- None reported.

## Interpretation and safeguards

- `target` and `phase_label` are labels/metadata, not candidate predictors.
- `normal_message_count` and `attack_message_count` are ground-truth-derived; exclude them from predictors.
- Exclude IDs, timestamps, scenario, window indices, and configured attack rate from model predictors.
- Review `unique_tcp_streams`: it may reveal whether the extra flood publisher was present.
- Duplicate predictor vectors need review before splitting; identical rows must not be scattered across train and test.
- Mixed windows remain mixed. Define their treatment before supervised binary classification.
- Correlation values are descriptive only and do not establish causation.
- Passing this audit does not establish generalization, unseen-attack detection, or model performance.

## Output files

- `data/processed/combined_pcap_features_v2.csv`
- `reports/combined_feature_dataset_audit_v2.csv`
- `reports/combined_feature_dataset_audit_v2.json`
- `reports/combined_feature_dataset_audit_v2.md`

Source feature CSV files were not modified.
