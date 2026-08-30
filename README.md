# Pitwall

Uncertainty-aware race strategy decision support from public F1 timing data. Case study: the 2024 Formula 1 São Paulo Grand Prix.

Status: Phase 0, data audit. There is no decision engine yet.

Research question

When conditions are changing and future weather and neutralization are unknown, which action should a race engineer prefer, and how confident can that preference be given only what was knowable at the time?

The project separates decision quality from outcome quality. A defensible decision can produce a poor result and a poor one can be rescued by luck. It is allowed to conclude that the actions taken in this race were right, wrong, or indistinguishable under the available uncertainty.

What Phase 0 found

The audit ran against the real session and changed the design three times.

Rain intensity does not exist in public data. FastF1's rainfall channel is boolean, once a minute, from one sensor. Weather state is now a derived index measured from the field's clean-air pace.
That index is unobservable at the decision points. It is estimable on 58 of 69 laps; the eleven gaps are all neutralized laps, including the window this project exists to study. So it becomes a state propagated across neutralizations with growing uncertainty, not a measurement.
Pit loss must be modelled on session time. On lap 28 four cars pitted under three different neutralization regimes, two of them entering the pit lane with under four seconds of VSC left.

Full measurements, limitations and the per-component feasibility verdict: docs/DATA_AUDIT.md.

Method commitments

Fixed before implementation.

No future information. Decisions at time t use [0, t] only, through a temporal gate rather than a raw session.
Compound choice is in the action space: stay out, fresh intermediates, full wets.
Utility is E[points] - λ·CVaR₀.₂(shortfall), one risk parameter, reported across a grid.
Common random numbers: one scenario per replication, evaluated against every action.
Confidence is the fraction of replications in which an action maximises utility, reported with its Monte Carlo standard error.
São Paulo 2024 is held out of every prior and calibration step.
docs/PREREGISTRATION.md freezes the methodology before any recommendation for this race is generated.
Data

FastF1 only, free and public. No proprietary or team-internal data is used and none is synthesised. Tyre temperatures and pressures, fuel loads, brake temperatures, team forecasts and rain radar are unavailable and are never approximated.

Running it
bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m pytest -m "not network"     # 57 offline tests
python scripts/run_data_audit.py      # first run downloads and caches the session
python -m pytest                      # adds the integration tests

Outputs land in data/audit/. docs/DATA_AUDIT.md is generated from them, with interpretation injected from docs/audit_narrative.md so reruns refresh the tables without overwriting the analysis.

Structure
config/     configuration and the variable registry
src/data/   session loading and audit functions
scripts/    Phase 0 entry point
tests/      offline and integration tests
docs/       data audit, narrative, AI assistance disclosure
Limitations

Public telemetry only. Weather from one trackside sensor with no intensity or spatial resolution. Tyre state unobservable. Fuel effect assumed. Counterfactual outcomes can never be verified, so validation targets component calibration, forced-scenario replay fidelity, robustness of the action ranking, and comparison against naive baselines. Full list in the data audit.

Roadmap

Phase 0 audit (done) → 0.5 multi-race corpus → 1 acquisition → 2 race reconstruction → 3 features and decision windows → 3.5 field simulator and replay fidelity → 4 pit loss and pace → 5 uncertainty, then preregistration → 6 Monte Carlo engine → 7 replay → 8 counterfactuals → 9 validation → 10 visualization → 11 case study.

Licence

MIT, see LICENSE. Contains no F1 data; sessions are fetched at runtime via FastF1. Unofficial, not associated with Formula 1, the FIA or any team. AI assistance disclosed in docs/AI_ASSISTANCE.md
