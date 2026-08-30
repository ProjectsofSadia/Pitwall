# data/

Nothing in this directory is committed. Everything here is regenerable from
`scripts/run_data_audit.py` plus the FastF1 API.

## cache/

FastF1's on-disk cache. Populated on first run and reused afterwards. Caching is
a correctness requirement, not a speed optimisation: without it every rerun hits
the live timing API, and upstream revisions to the data would silently change
results between runs of the same code.

To force a fresh download, delete the contents of this directory. To guarantee
that a run uses only cached data, pass `--offline`.

## audit/

Phase 0 outputs. Written fresh on every run.

| File | Contents |
| --- | --- |
| `session_columns.csv` | Per-column dtype, missingness, cardinality, sample values for every audited table |
| `missingness.csv` | Null counts and fractions, all tables |
| `expected_columns.csv` | Presence check for every column the design assumes exists |
| `driver_lap_counts.csv` | Per-driver lap counts, accuracy fractions, stints, compounds |
| `lap_track_status_distribution.csv` | Distribution of the concatenated per-lap `TrackStatus` string |
| `track_status.csv` | Decoded track status transitions with durations and approximate lap |
| `session_status.csv` | Session start, suspension, restart, finish |
| `race_control_messages.csv` | Full unaltered race control log |
| `race_control_summary.csv` | Message counts by category, flag and scope |
| `weather_audit.csv` | Per-channel summary statistics and missingness |
| `weather_sampling.csv` | Measured sampling interval distribution |
| `gaps.csv` | Reconstructed same-lap gaps from line-crossing session times |
| `gap_diagnostics.csv` | Per-lap validity statistics for the gap reconstruction |
| `wetness_feasibility.csv` | Cumulative eligibility counts for the wetness index, lap by lap |
| `pit_events.csv` | Every stop with track status at entry and exit |
| `stationary_probe.csv` | Whether a stationary segment is detectable in car telemetry |
| `telemetry_audit.csv` | Channel availability and measured sample rate |
| `data_classification.csv` | Variable registry resolved against the loaded session |
| `run_manifest.json` | Provenance: library versions, resolved session, config hashes, row counts |
| `audit_run.log` | Full run log including every warning |

`run_manifest.json` is the file to keep if you keep only one. It ties any figure
or table produced from a run to exact library versions, the resolved session and
the SHA-256 of the configuration that produced it.
