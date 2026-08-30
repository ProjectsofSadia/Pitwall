# AI assistance disclosure

This project was developed with the assistance of a large language model
(Anthropic Claude) used as a pair-programming and methodology-review tool.

## What the model was used for

- Critique of the initial project specification, including the arguments that
  led to replacing the FastF1 rainfall boolean with a derived field wetness
  index, replacing a four-weight utility function with expected points minus a
  CVaR penalty, and adopting a common-random-number Monte Carlo design.
- Drafting the Phase 0 audit modules, the runner script and the test suite.
- Diagnosing the timestamp-convention defect described in
  `TEMPORAL_INTEGRITY.md` and in the git history.

## What the author is responsible for

- Every methodological decision, including all six commitments listed in the
  README, and the choice to keep results that contradict the expected outcome.
- All execution against real data. No result in this repository was produced by
  the model; all audit outputs come from local runs against the FastF1 API.
- Verification of the findings and the final interpretation.

## Verification practice

Claims about FastF1 behaviour in this repository were checked against the
installed library source rather than against documentation or model recall.
Two examples that changed the code:

- `race_control_messages['Time']` is absolute `datetime64[ns]`, built with
  `to_datetime(entry['Utc'])`, while `weather_data`, `track_status` and
  `session_status` use `to_timedelta`. The initial implementation assumed one
  convention across all event tables and crashed on the real session.
- `Laps.TrackStatus` is a concatenation of every status code observed during
  the lap, not a single code, and `IsAccurate` already excludes safety car,
  VSC and red-flag laps.

Both defects were found by running against real data, not by review. The test
suite now contains regression tests for both.
