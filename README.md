# pitwall

Uncertainty-aware race strategy decision engine for the 2024 Formula 1 Sao Paulo Grand Prix.

**Status: Phase 0 (data audit). No decision engine exists yet.** This repository
currently contains a feasibility audit and nothing else. The sections describing
the eventual system are stated as intent, not as implemented behaviour.

## Research question

When conditions are changing rapidly, and future weather and race
neutralization are uncertain, what action should a race engineer prefer, and how
confident can that preference be given only the information available at the moment of the decision?

The project separates two things that are usually conflated: whether a decision
was good given what was knowable at the time, and whether it produced a good
outcome. A defensible decision can produce a poor result, and a poor decision can
be rescued by luck. The system is built to distinguish the two, and is permitted
to conclude that the actions taken in this race were wrong, right, or
indistinguishable under the available uncertainty.

## Method commitments

These were fixed before implementation and constrain everything downstream.

1. **No future information.** A decision evaluated at session time `t` may use
   observations from `[0, t]` only. From Phase 2, model and strategy code
   receives a `TemporalGate`, never a FastF1 session.
2. **Rainfall is not the weather state.** FastF1 exposes rainfall as a boolean at
   roughly one sample per minute, with no intensity. The weather state is instead
   a derived field wetness index `W(t)` measured in seconds per lap from
   clean-air field pace. Phase 0 tests whether the eligible sample supports it.
3. **Compound choice is in the action space from the start**: stay out, pit for
   fresh intermediates, pit for full wets.
4. **Utility is points, not seconds.**
   `U(a) = E[points(a)] - lambda * CVaR_alpha(points shortfall)`, with a single
   risk-aversion parameter `lambda` reported across a grid, and `alpha = 0.20`.
5. **Common random numbers.** One scenario draw per Monte Carlo replication,
   evaluated against every action, so actions are compared pairwise.
6. **Confidence has a definition.** `Confidence(a)` is the fraction of
   replications in which `a` maximises `U`, reported with its Monte Carlo
   standard error and a 95 percent interval.
7. **Sao Paulo 2024 is held out** of every prior, calibration and model-selection
   step that uses the multi-race corpus.
8. **Preregistration.** `docs/PREREGISTRATION.md` freezes the methodology before
   any recommendation for this race is generated. Later changes are recorded as
   amendments.

## Data sources

FastF1 (F1 live timing API), public and free. Lap timing, track status, race
control messages, weather, car and position telemetry, session results. No
proprietary, paid or team-internal data is used, and none is synthesised.

Tyre temperatures, tyre pressures, fuel loads, brake temperatures, team weather
forecasts and rain radar are not available and are never approximated.

## How to run Phase 0

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m pytest -m "not network"        # 33 offline tests
python scripts/run_data_audit.py         # first run downloads and caches the session
python -m pytest                         # includes the network tests
```

Outputs land in `data/audit/` as CSV plus `run_manifest.json`, and
`docs/DATA_AUDIT.md` is generated from them. Interpretation lives in
`docs/audit_narrative.md`, which is hand-written and injected into the generated
document at build time, so rerunning the audit refreshes the tables without
destroying the analysis.

## Phase 0 findings

The audit ran against the 2024 São Paulo Grand Prix race session and produced
three results that changed the design.

**Rain intensity does not exist in the public data.** The FastF1 `Rainfall`
channel is boolean, one sample per minute, from a single trackside sensor. This
was measured, not assumed, and it is why the weather state is a derived field
wetness index rather than a rainfall variable.

**The wetness index is unobservable at exactly the moments the project studies.**
W(t) is estimable on 58 of 69 laps. The eleven exceptions are neutralized laps,
and the first block of them (laps 27 to 33) contains the decision this project
exists to analyse. Under safety car and VSC drivers run to a delta time, so
green-flag field pace cannot be measured while there is no green flag. W(t) is
therefore reformulated as a filtered state propagated across neutralizations
with growing uncertainty, rather than a per-lap measurement. The consequence is
that uncertainty at the pivotal decision is large, which is the correct answer
rather than a deficiency.

**Pit loss must be modelled on session time, not lap number.** On lap 28, four
cars pitted under three different neutralization regimes: two entered the pit
lane with under four seconds of VSC remaining and exited under green, two
entered fully under green seconds later. Pit lane duration itself is close to
invariant across regimes (green 26.4 s, VSC 26.0 s, VSC-ending 25.2 s), so the
neutralization discount lives in the field rather than in the stop, and net loss
must be computed against a contemporaneous field reference.

Full detail, including the per-component feasibility verdict, is in
`docs/DATA_AUDIT.md`.

## Repository structure

```
config/     config.yaml and the variable registry (the audit specification)
data/       FastF1 cache and generated audit tables (both gitignored)
src/data/   session loading and audit functions
scripts/    Phase 0 entry point
tests/      offline unit and pipeline tests, plus network-marked integration tests
docs/       generated data audit
```

## Limitations

Stated up front rather than at the end. Public telemetry only. Weather comes
from a single trackside sensor with no spatial or intensity resolution. Tyre
state is not observable. Fuel effect is an assumption, not a measurement.
Counterfactual outcomes can never be verified, so the validation strategy
targets what can be checked: component calibration against ground truth,
forced-scenario replay fidelity, robustness of the action ranking under
perturbation, and comparison against naive baseline policies.

## Roadmap

Phase 0 data audit (current). Phase 0.5 multi-race corpus. Phase 1 repository and
acquisition. Phase 2 race reconstruction behind the temporal gate. Phase 3
features and decision windows. Phase 3.5 field simulator and forced-scenario
replay fidelity. Phase 4 pit loss and pace models. Phase 5 weather and
neutralization uncertainty, then preregistration. Phase 6 Monte Carlo decision
engine. Phase 7 replay. Phase 8 counterfactuals. Phase 9 validation, sensitivity,
ablation. Phase 10 visualization. Phase 11 documentation and case study.

## Licence

MIT, see `LICENSE`. This repository contains no Formula 1 data; session data is
retrieved at runtime from the F1 live timing API via FastF1. Unofficial project,
not associated with Formula 1, the FIA or any team.

## AI assistance

See `docs/AI_ASSISTANCE.md`.
