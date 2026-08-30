"""End-to-end smoke test of the Phase 0 pipeline, offline.

Runs ``run_audit`` against a synthetic session object with the same table shapes
FastF1 produces. This exists because the expensive failure mode is not a wrong
number, it is the audit crashing on row 200 of a real session after a ten-minute
download. Every output file and the generated document are exercised here with
no network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_data_audit as R  # noqa: E402
from src.data.fastf1_loader import LoadReport  # noqa: E402


def _td(x):
    return pd.to_timedelta(x, unit="s")


class FakeSession:
    """Minimal stand-in with the attributes run_audit reads."""

    def __init__(self):
        rows = []
        base_date = pd.to_datetime("2024-11-03 17:00:00")
        for lap in range(1, 6):
            for i, drv in enumerate(["VER", "NOR", "OCO"]):
                t = 90.0 * lap + 2.0 * i
                pit_in = _td(t) if (drv == "NOR" and lap == 3) else pd.NaT
                pit_out = _td(t - 60.0) if (drv == "NOR" and lap == 4) else pd.NaT
                rows.append(
                    dict(
                        Time=_td(t),
                        Driver=drv,
                        DriverNumber=str(i + 1),
                        LapTime=_td(90.0 + i),
                        LapNumber=float(lap),
                        Stint=1.0,
                        PitOutTime=pit_out,
                        PitInTime=pit_in,
                        Sector1Time=_td(30.0),
                        Sector2Time=_td(30.0),
                        Sector3Time=_td(30.0),
                        Sector1SessionTime=_td(t - 60.0),
                        Sector2SessionTime=_td(t - 30.0),
                        Sector3SessionTime=_td(t),
                        SpeedI1=250.0,
                        SpeedI2=260.0,
                        SpeedFL=280.0,
                        SpeedST=300.0,
                        IsPersonalBest=False,
                        Compound="INTERMEDIATE",
                        TyreLife=float(lap),
                        FreshTyre=True,
                        Team=f"Team{i}",
                        LapStartTime=_td(t - 90.0),
                        LapStartDate=base_date + pd.to_timedelta(t - 90.0, unit="s"),
                        TrackStatus="1",
                        Position=float(i + 1),
                        Deleted=False,
                        DeletedReason="",
                        FastF1Generated=False,
                        IsAccurate=True,
                    )
                )
        self.laps = pd.DataFrame(rows)

        self.weather_data = pd.DataFrame(
            {
                "Time": [_td(60.0 * i) for i in range(10)],
                "AirTemp": [21.0] * 10,
                "Humidity": [80.0] * 10,
                "Pressure": [1010.0] * 10,
                "Rainfall": [False] * 5 + [True] * 5,
                "TrackTemp": [25.0] * 10,
                "WindDirection": [180] * 10,
                "WindSpeed": [2.0] * 10,
            }
        )
        self.track_status = pd.DataFrame(
            {
                "Time": [_td(0.0), _td(250.0), _td(300.0), _td(320.0)],
                "Status": ["1", "6", "7", "1"],
                "Message": ["AllClear", "VSCDeployed", "VSCEnding", "AllClear"],
            }
        )
        # race_control_messages['Time'] is ABSOLUTE datetime64, not timedelta.
        # FastF1 builds it with to_datetime(entry['Utc']) while every other
        # event table uses to_timedelta. An earlier version of this fake used
        # timedelta here, which is why the offline suite passed while the real
        # session raised TypeError. The fake must match the API, not the
        # assumption under test.
        self.t0_date = base_date
        self.race_control_messages = pd.DataFrame(
            {
                "Time": [
                    base_date + pd.Timedelta(seconds=10.0),
                    base_date + pd.Timedelta(seconds=250.0),
                ],
                "Category": ["Flag", "Other"],
                "Message": ["GREEN LIGHT", "VIRTUAL SAFETY CAR DEPLOYED"],
                "Flag": ["GREEN", None],
                "Scope": ["Track", None],
                "Lap": [1, 3],
            }
        )
        self.session_status = pd.DataFrame(
            {"Time": [_td(0.0), _td(500.0)], "Status": ["Started", "Finished"]}
        )
        self.results = pd.DataFrame(
            {"Abbreviation": ["VER", "NOR", "OCO"], "Position": [1.0, 2.0, 3.0]}
        )

        t = pd.Series([i * 0.24 for i in range(2200)])
        self.car_data = {
            str(i + 1): pd.DataFrame(
                {
                    "Time": pd.to_timedelta(t, unit="s"),
                    "Date": base_date + pd.to_timedelta(t, unit="s"),
                    "Speed": [0.0 if 275 < v < 280 else 200.0 for v in t],
                    "RPM": [10000.0] * len(t),
                    "nGear": [5] * len(t),
                    "Throttle": [80.0] * len(t),
                    "Brake": [False] * len(t),
                    "DRS": [0] * len(t),
                }
            )
            for i in range(3)
        }
        self.pos_data = {
            str(i + 1): pd.DataFrame(
                {
                    "Time": pd.to_timedelta(t, unit="s"),
                    "Date": base_date + pd.to_timedelta(t, unit="s"),
                    "Status": ["OnTrack"] * len(t),
                    "X": [100] * len(t),
                    "Y": [200] * len(t),
                    "Z": [0] * len(t),
                }
            )
            for i in range(3)
        }


@pytest.fixture
def artefacts(tmp_path):
    cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    registry = yaml.safe_load(
        (REPO_ROOT / "config" / "variable_registry.yaml").read_text()
    )
    audit_dir, docs_dir = tmp_path / "audit", tmp_path / "docs"
    audit_dir.mkdir()
    docs_dir.mkdir()
    rc = R.run_audit(
        session=FakeSession(),
        report=LoadReport(requested={"spec": "synthetic"}),
        cfg=cfg,
        registry=registry,
        audit_dir=audit_dir,
        docs_dir=docs_dir,
        cfg_path=REPO_ROOT / "config" / "config.yaml",
        reg_path=REPO_ROOT / "config" / "variable_registry.yaml",
    )
    return rc, audit_dir, docs_dir


def test_pipeline_completes_and_writes_every_declared_output(artefacts):
    rc, audit_dir, _ = artefacts
    assert rc == 0
    for name in (
        "session_columns.csv",
        "missingness.csv",
        "expected_columns.csv",
        "driver_lap_counts.csv",
        "lap_track_status_distribution.csv",
        "track_status.csv",
        "session_status.csv",
        "race_control_messages.csv",
        "race_control_summary.csv",
        "weather_audit.csv",
        "weather_sampling.csv",
        "gaps.csv",
        "gap_diagnostics.csv",
        "wetness_feasibility.csv",
        "pit_events.csv",
        "stationary_probe.csv",
        "telemetry_audit.csv",
        "data_classification.csv",
        "run_manifest.json",
    ):
        assert (audit_dir / name).exists(), f"missing output: {name}"


def test_manifest_records_provenance_and_registry_state(artefacts):
    _, audit_dir, _ = artefacts
    manifest = json.loads((audit_dir / "run_manifest.json").read_text())
    assert manifest["config_sha256"] and manifest["registry_sha256"]
    assert "rows_written" in manifest
    assert manifest["rainfall_facts"]["rainfall_is_bool_dtype"] is True


def test_generated_document_has_every_required_section(artefacts):
    _, _, docs_dir = artefacts
    doc = (docs_dir / "DATA_AUDIT.md").read_text()
    for heading in (
        "## Purpose",
        "## FastF1 Environment",
        "## Session Overview",
        "## Lap Timing Data",
        "## Track Status",
        "## Race Control",
        "## Weather",
        "## Telemetry",
        "## Position Data",
        "## Pit Stop Observability",
        "## Gap Reconstruction",
        "## Wetness Index Feasibility",
        "## Temporal Resolution",
        "## Missingness and Reliability",
        "## Data Classification",
        "## Known Limitations",
        "## Implications for Methodology",
        "## Phase 0 Verdict",
    ):
        assert heading in doc, f"missing section: {heading}"
    assert "NARRATIVE" in doc, "interpretive sections must be flagged, not invented"


def test_registry_resolves_against_a_real_session_shape(artefacts):
    """Declared-OBSERVED variables must resolve without mismatch on a session
    that has the documented FastF1 schema. A mismatch here means the registry
    disagrees with FastF1, not with this particular race."""
    _, audit_dir, _ = artefacts
    cls = pd.read_csv(audit_dir / "data_classification.csv")
    observed = cls[cls["classification"] == "OBSERVED"]
    bad = observed[observed["status"] != "OK"]
    assert bad.empty, bad[["variable", "status"]].to_string()


def test_fake_session_matches_the_real_fastf1_timestamp_conventions():
    """Lock the fake to the API it stands in for.

    FastF1 parses race control message timestamps from the absolute `Utc` field
    with to_datetime, and every other event table with to_timedelta. When this
    fake disagreed with that, the offline suite passed and the real session
    crashed. This test makes the divergence impossible to reintroduce silently.
    """
    s = FakeSession()
    assert pd.api.types.is_datetime64_any_dtype(s.race_control_messages["Time"])
    for table in ("weather_data", "track_status", "session_status"):
        col = getattr(s, table)["Time"]
        assert pd.api.types.is_timedelta64_dtype(col), f"{table}.Time"
    assert pd.api.types.is_timedelta64_dtype(s.laps["Time"])


def test_race_control_lands_on_the_session_time_axis(artefacts):
    """Absolute message timestamps must become session-relative seconds, on the
    same axis as track status, not epoch seconds."""
    _, audit_dir, _ = artefacts
    rcm = pd.read_csv(audit_dir / "race_control_messages.csv")
    assert "time_s" in rcm.columns
    assert rcm["time_s"].tolist() == pytest.approx([10.0, 250.0])

    ts = pd.read_csv(audit_dir / "track_status.csv")
    assert ts["time_s"].max() < 10_000, "track status is not on the session axis"
    assert rcm["time_s"].max() < 10_000, "race control is on the wrong time axis"

    manifest = json.loads((audit_dir / "run_manifest.json").read_text())
    assert manifest["race_control_time_axis_resolved"] is True
    assert manifest["session_time_reference_t0_date"]


def test_narrative_sections_are_injected_not_overwritten(tmp_path):
    """Rerunning the audit must refresh the tables without destroying the
    hand-written analysis. The prose lives in docs/audit_narrative.md and is
    substituted into the generated document at build time."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "audit_narrative.md").write_text(
        "# Audit narrative\n\n"
        "## Known Limitations\n\nThe weather state is unobservable at the "
        "decision points.\n\n"
        "## Phase 0 Verdict\n\nPasses with one amendment.\n",
        encoding="utf-8",
    )
    sections = R.load_narrative(docs)
    assert set(sections) == {"Known Limitations", "Phase 0 Verdict"}
    assert "unobservable" in sections["Known Limitations"]

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    registry = yaml.safe_load(
        (REPO_ROOT / "config" / "variable_registry.yaml").read_text()
    )
    R.run_audit(
        session=FakeSession(),
        report=LoadReport(requested={"spec": "synthetic"}),
        cfg=cfg,
        registry=registry,
        audit_dir=audit_dir,
        docs_dir=docs,
        cfg_path=REPO_ROOT / "config" / "config.yaml",
        reg_path=REPO_ROOT / "config" / "variable_registry.yaml",
    )
    doc = (docs / "DATA_AUDIT.md").read_text()
    assert "unobservable at the\ndecision points" in doc or "unobservable" in doc
    assert "Passes with one amendment" in doc
    # Sections with no narrative entry must still show the placeholder.
    assert "NARRATIVE" in doc


def test_shipped_narrative_covers_every_generated_section():
    """A heading typo in audit_narrative.md would silently leave a placeholder
    in the published document. Fail here instead."""
    sections = R.load_narrative(REPO_ROOT / "docs")
    required = {
        "Lap Timing Data", "Track Status", "Race Control", "Weather",
        "Telemetry", "Position Data", "Pit Stop Observability",
        "Gap Reconstruction", "Wetness Index Feasibility",
        "Temporal Resolution", "Missingness and Reliability",
        "Data Classification", "Known Limitations",
        "Implications for Methodology", "Phase 0 Verdict",
    }
    missing = required - set(sections)
    assert not missing, f"narrative missing sections: {sorted(missing)}"
    empty = {k for k in required if not sections.get(k, "").strip()}
    assert not empty, f"narrative sections are empty: {sorted(empty)}"
