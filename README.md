# Pitwall

**Uncertainty-Aware Race Strategy Decision Support for Formula One**

Pitwall is a research-oriented motorsport engineering project for evaluating race-strategy decisions under incomplete information. The flagship case study reconstructs the 2024 São Paulo Grand Prix and asks a specific engineering question:

> Given only the information available at decision time, should a race engineer stay out, pit for fresh intermediates, or switch to full wets?

The system is designed around strict temporal causality, probabilistic simulation, uncertainty quantification, and counterfactual evaluation rather than hindsight-based race analysis.

---

## Engineering Objective

Race strategy decisions are made before future weather, neutralizations, traffic evolution, and competitor actions are known. Pitwall models that decision environment explicitly.

The project separates:

- **Decision quality** — whether an action was defensible using information available at the time
- **Outcome quality** — what happened afterward
- **Uncertainty** — how sensitive the preferred action is to plausible future race states

A strategy can therefore produce a poor result while still being rational ex ante, or succeed because of events that could not reasonably have been predicted.

---

## Current Status

**Phase 0 — Data and Method Feasibility: Complete**

The first stage audited the public-data pipeline before any strategy model was allowed to generate recommendations.

The audit validated:

- FastF1 timing and telemetry availability
- session-time race-state reconstruction
- weather-channel limitations
- pit-event reconstruction
- track-status timing
- clean-air wetness-state feasibility
- gap reconstruction
- data provenance and variable classification

See [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) for the detailed feasibility assessment.

---

## Key Findings from the Data Audit

### Public rainfall data is not rain intensity

FastF1 exposes rainfall as a boolean channel sampled approximately once per minute from a trackside source. It cannot support a continuous rain-intensity model.

Pitwall therefore derives a **field wetness state** primarily from observed race performance rather than treating the rainfall flag as a physical intensity measurement.

### Wetness cannot always be directly observed

Clean-air field pace can estimate wetness during much of the race, but neutralized periods remove the observations required for direct inference. During these windows the state must therefore be propagated forward with increasing uncertainty.

### Pit loss is time-dependent

Pit-loss evaluation cannot be treated as a constant lap-level penalty. At São Paulo, cars entered the pit lane under materially different neutralization conditions within the same lap. Pit-loss modeling therefore operates on **session time**, with track-status transitions resolved explicitly.

These findings changed the architecture before model implementation rather than being patched into the system afterward.

---

## Decision Framework

At a decision time `t`, the candidate actions are:

1. **Stay out** on the current intermediate tyre
2. **Pit for fresh intermediates**
3. **Pit for full wet tyres**

Only information available in the interval `[0, t]` is accessible to the decision system. Future observations from the recorded race are prohibited. A dedicated temporal gate will enforce this constraint throughout the pipeline.

---

## Risk-Aware Utility

Candidate strategies are evaluated using expected championship value together with downside risk:

`U(a) = E[points(a)] - λ · CVaR₀.₂(shortfall(a))`

Candidate actions are evaluated across multiple values of risk aversion using paired Monte Carlo simulation.

Pitwall will report:

- expected outcome distributions
- probability each action is optimal
- Monte Carlo standard error
- confidence intervals
- ranking stability under parameter uncertainty
- sensitivity to risk preference

---

## Simulation Methodology

Each Monte Carlo replication represents one plausible future race state. The same realization of weather evolution, neutralization timing, competitor behavior, pace variation, pit stationary time, and traffic uncertainty is evaluated against every candidate strategy.

This **common-random-numbers (CRN)** design enables paired comparisons between actions and reduces Monte Carlo noise in strategy ranking.

---

## Validation Philosophy

São Paulo 2024 is the flagship case study, not the calibration dataset. The race is held out from prior fitting, model selection, calibration, and parameter tuning.

Historical races will form the calibration corpus. Before the decision engine is allowed to evaluate the São Paulo strategy window, the methodology will be frozen in [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).

Validation will focus on:

- forced-scenario replay fidelity
- pace-model calibration
- neutralization-probability calibration
- probabilistic forecast quality
- Monte Carlo convergence and seed stability
- comparison with naive strategy baselines
- sensitivity analysis
- ablation testing
- counterfactual regret

Agreement with the strategy chosen by an actual Formula One team is **not** treated as model accuracy.

---

## Data Integrity

Pitwall uses public FastF1 data only. The project does **not** claim access to tyre carcass or surface temperatures, tyre pressures, fuel loads, brake temperatures, proprietary tyre models, team weather forecasts, radar, internal strategy tools, or engineering sensor channels unavailable publicly.

Unavailable measurements are documented rather than fabricated.

Variables are classified as:

- `OBSERVED`
- `DERIVED`
- `ESTIMATED`
- `ASSUMED`
- `SIMULATED`

This provenance is maintained throughout the modeling pipeline.

---

## Architecture

```text
FastF1 Session Data
        |
        v
Data Validation + Provenance Audit
        |
        v
Temporal Gate (information <= t only)
        |
        v
Chronological Race Reconstruction
        |
        +--> Wetness Model
        +--> Pace Model
        +--> Pit-Loss Model
        +--> Neutralization Model
                    |
                    v
              Field Simulator
                    |
                    v
         Monte Carlo Decision Engine
                    |
                    v
       Strategy Utility + Risk + Uncertainty
```

---

## Repository Structure

```text
Pitwall/
├── config/          # model and audit configuration
├── data/            # generated audit artefacts and cache
├── docs/            # methodology and research documentation
├── scripts/         # reproducible pipeline entry points
├── src/             # application and modeling code
├── tests/           # unit, regression and integration tests
├── pyproject.toml
└── README.md
```

Planned modeling modules include dedicated temporal, modeling, simulation, and evaluation components for wetness, pace, pit loss, neutralization risk, field simulation, overtaking, restart behavior, and counterfactual regret.

---

## Reproducing Phase 0

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"

python -m pytest -m "not network"
python scripts\run_data_audit.py
python -m pytest
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

python -m pytest -m "not network"
python scripts/run_data_audit.py
python -m pytest
```

Generated audit artefacts are written to `data/audit/`. The data-audit report is regenerated from those outputs so measurements remain tied to the executable pipeline.

---

## Research Roadmap

### Completed

- [x] Phase 0 — data feasibility and provenance audit
- [x] automated audit validation
- [x] race-control time-reference validation
- [x] wetness feasibility analysis
- [x] pit-event reconstruction audit

### In Development

- [ ] multi-race historical calibration corpus
- [ ] temporal information gate
- [ ] chronological race reconstruction
- [ ] wetness-state estimator
- [ ] pace model
- [ ] pit-loss model
- [ ] neutralization hazard model
- [ ] field simulator
- [ ] forced-scenario replay validation
- [ ] methodology preregistration
- [ ] common-random-numbers Monte Carlo engine
- [ ] counterfactual strategy evaluation
- [ ] sensitivity and ablation analysis

---

## Design Principles

**No hindsight.** Future race observations cannot influence historical decisions.

**No synthetic proprietary telemetry.** Unavailable Formula One engineering channels remain unavailable.

**Simple models before complex models.** Additional complexity must improve out-of-sample performance or calibration.

**Uncertainty is part of the result.** A strategy recommendation without uncertainty is incomplete.

**Replay before counterfactuals.** The simulator must reproduce constrained historical scenarios before being trusted for alternative futures.

**Indeterminate is a valid conclusion.** The system is not required to prove that one historical strategy was correct.

---

## Limitations

Public Formula One telemetry cannot reproduce the information environment available to an actual race team. Important latent variables remain unavailable, including detailed tyre state, fuel mass, local rainfall distribution, radar forecasts, driver feedback, and proprietary vehicle models.

The project therefore evaluates **decision-making from public information**, not the full capability of an operational Formula One strategy group.

Counterfactual race outcomes are inherently unobservable. Validation consequently targets model calibration, replay fidelity, robustness, and comparative decision quality rather than claiming ground-truth counterfactual accuracy.

---

## Technology

**Python · FastF1 · pandas · NumPy · SciPy · scikit-learn · probabilistic simulation · Monte Carlo methods · statistical modeling · pytest**

MATLAB-based analysis is planned for selected simulation, sensitivity, and uncertainty-analysis workflows.

---

## Licence

MIT License.

Race-session data is retrieved at runtime through FastF1 and is not distributed with this repository.

Pitwall is an independent research project and is not affiliated with Formula 1, the FIA, or any Formula One team.
