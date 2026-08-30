"""Phase 0 tests.

The audit functions are pure with respect to dataframes, so almost everything
here runs offline against small synthetic frames whose correct answers are known
by construction. Only the tests marked ``network`` require the live timing API
or a warm cache.

Nothing here tests that pandas works. Each test targets a specific way the audit
could be silently wrong: a filter that is not actually cumulative, a gap that is
attributed to the wrong car, a weather sample from the future being joined to a
lap, or a registry claim that no longer matches the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import audit as A  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _td(seconds):
    return pd.to_timedelta(seconds, unit="s")


@pytest.fixture
def toy_laps() -> pd.DataFrame:
    """Three cars, two laps. Crossing times chosen so gaps are known exactly.

    Lap 1: A at 100 s, B at 102 s, C at 105.5 s  -> gaps 2.0 and 3.5
    Lap 2: A at 190 s, B at 193 s, C at 199.0 s  -> gaps 3.0 and 6.0
    C pits at the end of lap 1 and rejoins on lap 2.
    """
    rows = [
        dict(Driver="A", DriverNumber="1", LapNumber=1.0, Position=1.0, Time=_td(100.0),
             LapTime=_td(90.0), Compound="INTERMEDIATE", TyreLife=3.0, Stint=1.0,
             TrackStatus="1", IsAccurate=True, PitInTime=pd.NaT, PitOutTime=pd.NaT,
             LapStartTime=_td(10.0), Team="T1", FreshTyre=True),
        dict(Driver="B", DriverNumber="2", LapNumber=1.0, Position=2.0, Time=_td(102.0),
             LapTime=_td(91.0), Compound="INTERMEDIATE", TyreLife=3.0, Stint=1.0,
             TrackStatus="1", IsAccurate=True, PitInTime=pd.NaT, PitOutTime=pd.NaT,
             LapStartTime=_td(11.0), Team="T2", FreshTyre=True),
        dict(Driver="C", DriverNumber="3", LapNumber=1.0, Position=3.0, Time=_td(105.5),
             LapTime=_td(94.0), Compound="INTERMEDIATE", TyreLife=3.0, Stint=1.0,
             TrackStatus="1", IsAccurate=True, PitInTime=_td(104.0), PitOutTime=pd.NaT,
             LapStartTime=_td(11.5), Team="T3", FreshTyre=True),
        dict(Driver="A", DriverNumber="1", LapNumber=2.0, Position=1.0, Time=_td(190.0),
             LapTime=_td(90.0), Compound="INTERMEDIATE", TyreLife=4.0, Stint=1.0,
             TrackStatus="1", IsAccurate=True, PitInTime=pd.NaT, PitOutTime=pd.NaT,
             LapStartTime=_td(100.0), Team="T1", FreshTyre=True),
        dict(Driver="B", DriverNumber="2", LapNumber=2.0, Position=2.0, Time=_td(193.0),
             LapTime=_td(91.0), Compound="INTERMEDIATE", TyreLife=4.0, Stint=1.0,
             TrackStatus="14", IsAccurate=False, PitInTime=pd.NaT, PitOutTime=pd.NaT,
             LapStartTime=_td(102.0), Team="T2", FreshTyre=True),
        dict(Driver="C", DriverNumber="3", LapNumber=2.0, Position=3.0, Time=_td(199.0),
             LapTime=_td(93.5), Compound="WET", TyreLife=1.0, Stint=2.0,
             TrackStatus="1", IsAccurate=True, PitInTime=pd.NaT,
             PitOutTime=_td(126.0), LapStartTime=_td(105.5), Team="T3",
             FreshTyre=True),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def toy_track_status() -> pd.DataFrame:
    """Green, then VSC deployed at 110 s, VSC ending at 120 s, green at 125 s."""
    return pd.DataFrame(
        {
            "Time": [_td(0.0), _td(110.0), _td(120.0), _td(125.0)],
            "Status": ["1", "6", "7", "1"],
            "Message": ["AllClear", "VSCDeployed", "VSCEnding", "AllClear"],
        }
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_shipped_registry_is_valid():
    """The registry in the repo must parse and use only approved labels."""
    registry = yaml.safe_load(
        (REPO_ROOT / "config" / "variable_registry.yaml").read_text(encoding="utf-8")
    )
    assert A.validate_registry(registry) == []
    labels = {v["classification"] for v in registry["variables"]}
    assert labels <= A.CLASSIFICATIONS


def test_registry_rejects_unapproved_classification():
    bad = {"variables": [{"name": "x", "classification": "PROBABLY",
                          "resolver": "deferred", "description": "d",
                          "source": "s", "planned_use": "p"}]}
    problems = A.validate_registry(bad)
    assert any("PROBABLY" in p for p in problems)


def test_registry_rejects_duplicate_names():
    var = {"name": "x", "classification": "OBSERVED", "resolver": "deferred",
           "description": "d", "source": "s", "planned_use": "p"}
    problems = A.validate_registry({"variables": [var, dict(var)]})
    assert any("duplicate" in p for p in problems)


def test_every_classification_has_a_leakage_rule():
    """No approved classification may fall through to 'unclassified'."""
    for cls in A.CLASSIFICATIONS:
        assert A._leakage_risk(cls, "deferred") != "unclassified"


# --------------------------------------------------------------------------
# Column audits
# --------------------------------------------------------------------------


def test_expected_column_report_records_absence_without_raising(toy_laps):
    """A missing column is a finding, not an exception."""
    df = toy_laps.drop(columns=["Compound"])
    rep = A.expected_column_report(df, ("Compound", "Driver"), "laps")
    row = rep[rep["column"] == "Compound"].iloc[0]
    assert bool(row["present"]) is False
    assert row["null_frac"] == 1.0


def test_column_inventory_counts_nulls(toy_laps):
    df = toy_laps.copy()
    df.loc[0, "Compound"] = None
    inv = A.column_inventory(df, "laps")
    row = inv[inv["column"] == "Compound"].iloc[0]
    assert row["n_null"] == 1
    assert row["n_rows"] == len(df)


def test_per_lap_track_status_flags_mixed_laps(toy_laps):
    dist = A.per_lap_track_status_distribution(toy_laps)
    mixed = dist[dist["track_status_string"] == "14"]
    assert len(mixed) == 1
    assert bool(mixed.iloc[0]["is_pure_green"]) is False
    assert "SafetyCar" in mixed.iloc[0]["decoded_codes"]


# --------------------------------------------------------------------------
# Track status
# --------------------------------------------------------------------------


def test_track_status_decoding_and_durations(toy_track_status):
    out = A.audit_track_status(toy_track_status)
    assert list(out["decoded"]) == [
        "AllClear",
        "VSCDeployed",
        "VSCEnding",
        "AllClear",
    ]
    assert out.loc[1, "duration_s"] == pytest.approx(10.0)
    assert out.loc[1, "transition"] == "1->6"


def test_unknown_track_status_code_is_surfaced_not_swallowed():
    ts = pd.DataFrame({"Time": [_td(0.0)], "Status": ["9"], "Message": ["?"]})
    out = A.audit_track_status(ts)
    assert "UNKNOWN(9)" in out.loc[0, "decoded"]


def test_audit_track_status_handles_empty():
    assert len(A.audit_track_status(pd.DataFrame())) == 0


# --------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------


def test_gaps_match_constructed_values(toy_laps):
    gaps = A.compute_same_lap_gaps(toy_laps)
    lap1 = gaps[gaps["LapNumber"] == 1.0].set_index("Driver")
    assert np.isnan(lap1.loc["A", "gap_ahead_s"])  # leader has no car ahead
    assert lap1.loc["B", "gap_ahead_s"] == pytest.approx(2.0)
    assert lap1.loc["C", "gap_ahead_s"] == pytest.approx(3.5)
    assert lap1.loc["A", "gap_behind_s"] == pytest.approx(2.0)
    assert lap1.loc["B", "driver_ahead"] == "A"


def test_gaps_flag_order_position_disagreement():
    """Crossing order disagreeing with classification must be visible."""
    laps = pd.DataFrame(
        {
            "Driver": ["A", "B"],
            "LapNumber": [1.0, 1.0],
            "Position": [2.0, 1.0],  # B classified ahead but crosses later
            "Time": [_td(100.0), _td(101.0)],
        }
    )
    gaps = A.compute_same_lap_gaps(laps)
    assert not gaps["order_matches_position"].all()


def test_gap_diagnostics_reports_cars_not_on_leader_lap(toy_laps):
    laps = toy_laps[~((toy_laps.Driver == "C") & (toy_laps.LapNumber == 2.0))]
    gaps = A.compute_same_lap_gaps(laps)
    diag = A.gap_diagnostics(gaps, toy_laps)
    lap2 = diag[diag["LapNumber"] == 2.0].iloc[0]
    assert lap2["n_cars_on_lap"] == 2
    assert lap2["cars_not_on_leader_lap"] == 1


def test_gaps_missing_required_columns_returns_empty():
    assert len(A.compute_same_lap_gaps(pd.DataFrame({"Driver": ["A"]}))) == 0


# --------------------------------------------------------------------------
# Wetness feasibility
# --------------------------------------------------------------------------


def test_wetness_filter_stages_are_cumulative(toy_laps):
    """Each stage may only shrink the sample. A non-monotone count means the
    filter is not actually cumulative and the diagnosis of where the sample
    collapses would be wrong."""
    gaps = A.compute_same_lap_gaps(toy_laps)
    wet = A.wetness_feasibility(
        toy_laps, gaps, ["INTERMEDIATE", "WET"], ["1"], [1.5]
    )
    stages = [
        "n_stage0_all_rows",
        "n_stage1_laptime",
        "n_stage2_is_accurate",
        "n_stage3_not_pit_lap",
        "n_stage4_green_lap",
        "n_stage5_wet_compound",
        "n_stage6_clean_air_1.5s",
    ]
    for _, row in wet.iterrows():
        values = [row[s] for s in stages]
        assert all(a >= b for a, b in zip(values, values[1:])), values


def test_wetness_excludes_pit_and_non_green_laps(toy_laps):
    gaps = A.compute_same_lap_gaps(toy_laps)
    wet = A.wetness_feasibility(
        toy_laps, gaps, ["INTERMEDIATE", "WET"], ["1"], [1.5]
    ).set_index("LapNumber")
    # Lap 1: C is a pit-in lap, so 3 -> 2 at stage 3.
    assert wet.loc[1.0, "n_stage2_is_accurate"] == 3
    assert wet.loc[1.0, "n_stage3_not_pit_lap"] == 2
    # Lap 2: B is inaccurate under safety car, C is a pit-out lap. Only A remains.
    assert wet.loc[2.0, "n_stage4_green_lap"] == 1


def test_wetness_counts_leader_as_clean_air(toy_laps):
    """The leader has no car ahead and NaN gap. Treating NaN as 'not clean' would
    silently drop the single most informative car from the wetness sample."""
    gaps = A.compute_same_lap_gaps(toy_laps)
    wet = A.wetness_feasibility(
        toy_laps, gaps, ["INTERMEDIATE", "WET"], ["1"], [10.0]
    ).set_index("LapNumber")
    # With a 10 s clean-air threshold nobody qualifies on gap, but the leader
    # must still be counted.
    assert wet.loc[1.0, "n_stage6_clean_air_10.0s"] >= 1


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------


def test_rainfall_dtype_is_measured_not_assumed():
    w = pd.DataFrame(
        {
            "Time": [_td(0.0), _td(60.0), _td(120.0)],
            "AirTemp": [21.0, 21.2, 21.1],
            "Rainfall": [False, True, True],
        }
    )
    _, sampling, facts = A.audit_weather(w)
    assert facts["rainfall_present"] is True
    assert facts["rainfall_is_bool_dtype"] is True
    assert facts["rainfall_n_unique"] == 2
    assert sampling.loc[0, "dt_median_s"] == pytest.approx(60.0)


def test_non_boolean_rainfall_would_be_detected():
    """If a future FastF1 release ships an intensity channel, the audit must not
    keep reporting it as boolean."""
    w = pd.DataFrame({"Time": [_td(0.0), _td(60.0)], "Rainfall": [0.0, 2.4]})
    _, _, facts = A.audit_weather(w)
    assert facts["rainfall_is_bool_dtype"] is False


def test_attach_conditions_never_uses_a_future_weather_sample():
    """The weather join must be nearest-preceding. A sample taken after the lap
    started is future information and must not be attached."""
    wet = pd.DataFrame({"LapNumber": [1.0, 2.0]})
    leader = pd.DataFrame(
        {"LapNumber": [1.0, 2.0], "leader_lap_start_s": [50.0, 150.0]}
    )
    weather = pd.DataFrame(
        {"Time": [_td(0.0), _td(100.0), _td(200.0)], "AirTemp": [10.0, 20.0, 30.0]}
    )
    out = A.attach_conditions(wet, weather, leader)
    # Lap 1 starts at 50 s: only the 0 s sample precedes it.
    assert out.loc[0, "weather_AirTemp"] == 10.0
    # Lap 2 starts at 150 s: the 100 s sample, never the 200 s one.
    assert out.loc[1, "weather_AirTemp"] == 20.0


# --------------------------------------------------------------------------
# Pit events
# --------------------------------------------------------------------------


def test_pit_event_pairing_and_regime_classification(toy_laps, toy_track_status):
    ts = A.audit_track_status(toy_track_status)
    events = A.extract_pit_events(toy_laps, ts)
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["driver"] == "C"
    assert ev["in_lap"] == 1.0 and ev["out_lap"] == 2.0
    assert ev["pit_lane_duration_s"] == pytest.approx(22.0)
    assert ev["compound_before"] == "INTERMEDIATE"
    assert ev["compound_after"] == "WET"


def test_pit_event_detects_status_change_during_the_stop(toy_laps, toy_track_status):
    """The stop enters at 104 s under green and exits at 126 s under green, but
    a VSC was deployed and ended in between. The audit must record the entry
    regime, the exit regime and the time to the next transition, because a
    lap-resolution model cannot distinguish these cases."""
    ts = A.audit_track_status(toy_track_status)
    ev = A.extract_pit_events(toy_laps, ts).iloc[0]
    assert ev["track_status_at_pit_in"] == "1"
    assert ev["track_status_at_pit_out"] == "1"
    assert ev["seconds_to_next_status_change_from_pit_in"] == pytest.approx(6.0)


def test_pit_events_missing_columns_returns_empty():
    assert len(A.extract_pit_events(pd.DataFrame({"Driver": ["A"]}), pd.DataFrame())) == 0


def test_stationary_probe_finds_the_constructed_halt():
    events = pd.DataFrame(
        [
            dict(driver="C", driver_number="3", in_lap=1.0, pit_in_s=100.0,
                 pit_out_s=120.0, pit_lane_duration_s=20.0)
        ]
    )
    t = np.arange(95.0, 125.0, 0.25)
    speed = np.where((t >= 105.0) & (t < 107.5), 0.0, 60.0)
    car = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "Date": pd.to_datetime("2024-11-03 17:00:00") + pd.to_timedelta(t, unit="s"),
            "Speed": speed,
        }
    )
    out = A.stationary_probe(events, lambda num: car, 5.0, 5)
    assert out.loc[0, "longest_below_run_samples"] == 10
    assert out.loc[0, "longest_below_run_s"] == pytest.approx(2.25, abs=0.3)


def test_stationary_probe_reports_missing_telemetry_as_error_not_zero():
    """Absent telemetry must not be indistinguishable from a zero-second stop."""
    events = pd.DataFrame(
        [dict(driver="C", driver_number="3", in_lap=1.0, pit_in_s=100.0,
              pit_out_s=120.0, pit_lane_duration_s=20.0)]
    )
    out = A.stationary_probe(events, lambda num: None, 5.0, 5)
    assert "error" in out.columns and out.loc[0, "error"]


# --------------------------------------------------------------------------
# Registry resolution
# --------------------------------------------------------------------------


def _ctx(**over):
    base = {
        "laps_columns": {"Compound": 0.0, "LapNumber": 0.0},
        "weather_columns": {"Rainfall": 0.0},
        "tables": {"track_status": 12},
        "car_data_channels": {"Speed"},
        "pos_data_channels": {"X"},
        "derived": {"gaps": {"available": True, "note": "ok", "resolution": "per lap"}},
        "weather_dt_median_s": 60.0,
        "telemetry_dt_median_ms": 240.0,
    }
    base.update(over)
    return base


def _var(**over):
    v = dict(name="compound", description="d", source="s", classification="OBSERVED",
             resolver="laps_column:Compound", planned_use="p", notes="")
    v.update(over)
    return v


def test_resolve_registry_marks_present_observed_as_ok():
    out = A.resolve_registry({"variables": [_var()]}, _ctx())
    assert bool(out.loc[0, "available"]) is True
    assert out.loc[0, "status"] == "OK"


def test_resolve_registry_flags_missing_observed_variable():
    """The specification cannot be quietly satisfied by data that is not there."""
    out = A.resolve_registry(
        {"variables": [_var(name="tyre_temp", resolver="laps_column:TyreTemp")]}, _ctx()
    )
    assert out.loc[0, "status"].startswith("MISMATCH")


def test_resolve_registry_flags_not_available_that_turns_up():
    out = A.resolve_registry(
        {"variables": [_var(name="Rainfall", classification="NOT_AVAILABLE",
                            resolver="absent")]},
        _ctx(),
    )
    assert out.loc[0, "status"].startswith("MISMATCH")


def test_resolve_registry_marks_deferred_variables():
    out = A.resolve_registry(
        {"variables": [_var(name="pit_loss", classification="ESTIMATED",
                            resolver="deferred")]},
        _ctx(),
    )
    assert out.loc[0, "available"] == "DEFERRED"
    assert "corpus must exclude" in out.loc[0, "leakage_risk"]


# --------------------------------------------------------------------------
# Integration (requires network or a warm cache)
# --------------------------------------------------------------------------


@pytest.mark.network
def test_session_loads_and_core_tables_are_populated():
    from src.data.fastf1_loader import SessionSpec, enable_cache, load_session

    cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    cache = enable_cache(REPO_ROOT / cfg["paths"]["cache_dir"])
    spec = SessionSpec(
        cfg["session"]["year"], cfg["session"]["grand_prix"], cfg["session"]["identifier"]
    )
    session, report = load_session(spec, cache, cfg["session"]["load"])

    assert len(session.laps) > 0
    assert report.resolved_event_name
    for table in ("weather_data", "track_status", "race_control_messages"):
        assert len(getattr(session, table)) > 0, f"{table} is empty"

    _, _, facts = A.audit_weather(pd.DataFrame(session.weather_data))
    assert facts["rainfall_present"], "Rainfall channel absent from weather data"


@pytest.mark.network
def test_audit_outputs_exist_after_a_run():
    """Run scripts/run_data_audit.py before this test."""
    cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    audit_dir = REPO_ROOT / cfg["paths"]["audit_dir"]
    for name in (
        "session_columns.csv",
        "missingness.csv",
        "weather_audit.csv",
        "track_status.csv",
        "race_control_messages.csv",
        "driver_lap_counts.csv",
        "pit_events.csv",
        "wetness_feasibility.csv",
        "gap_diagnostics.csv",
        "data_classification.csv",
        "run_manifest.json",
    ):
        assert (audit_dir / name).exists(), f"missing audit output: {name}"

    cls = pd.read_csv(audit_dir / "data_classification.csv")
    assert set(cls["classification"]) <= A.CLASSIFICATIONS


# --------------------------------------------------------------------------
# Timestamp conventions
#
# Regression tests for the Phase 0 execution failure: FastF1 parses
# race_control_messages['Time'] from the absolute 'Utc' field with to_datetime,
# while weather_data, track_status and session_status use to_timedelta. The
# original _seconds() assumed one convention across all event tables and raised
# TypeError on the real session.
# --------------------------------------------------------------------------


T0 = pd.Timestamp("2024-11-03 17:00:00")


def test_to_session_seconds_passes_timedelta_through_unchanged():
    s = pd.Series(pd.to_timedelta([0.0, 90.5, 181.0], unit="s"))
    assert A.to_session_seconds(s).tolist() == pytest.approx([0.0, 90.5, 181.0])


def test_datetime64_time_column_converts_with_a_reference():
    """Reproduces the real failure: a datetime64[ns] Time column."""
    s = pd.Series([T0, T0 + pd.Timedelta(seconds=90.5)])
    assert pd.api.types.is_datetime64_any_dtype(s)
    out = A.to_session_seconds(s, t0=T0, column="Time")
    assert out.tolist() == pytest.approx([0.0, 90.5])


def test_datetime64_without_a_reference_raises_a_named_error_not_typeerror():
    """The failure must be actionable and must never fall back to epoch seconds:
    epoch values sort and merge without complaint and would be silently wrong."""
    s = pd.Series([T0, T0 + pd.Timedelta(seconds=90.5)])
    with pytest.raises(A.TimestampReferenceError) as exc:
        A.to_session_seconds(s, t0=None, column="Time")
    assert "Time" in str(exc.value)
    assert "reference" in str(exc.value).lower()


def test_numeric_time_column_is_refused_rather_than_guessed():
    with pytest.raises(A.TimestampReferenceError):
        A.to_session_seconds(pd.Series([0.0, 90.5]), t0=T0, column="Time")


def test_datetime_and_timedelta_representations_agree_given_the_reference():
    """The equivalence the whole fix rests on: the same instants expressed
    absolutely and session-relative must map to identical session seconds."""
    offsets = [0.0, 12.25, 900.0, 3600.75]
    as_timedelta = pd.Series(pd.to_timedelta(offsets, unit="s"))
    as_datetime = pd.Series([T0 + pd.Timedelta(seconds=o) for o in offsets])
    left = A.to_session_seconds(as_timedelta)
    right = A.to_session_seconds(as_datetime, t0=T0)
    assert left.tolist() == pytest.approx(right.tolist())
    assert left.tolist() == pytest.approx(offsets)


def test_timezone_aware_timestamps_are_normalised_before_subtraction():
    s = pd.Series(
        [T0.tz_localize("UTC"), (T0 + pd.Timedelta(seconds=60)).tz_localize("UTC")]
    )
    assert A.to_session_seconds(s, t0=T0).tolist() == pytest.approx([0.0, 60.0])


def test_resolve_t0_prefers_the_session_attribute():
    class S:
        t0_date = T0

    assert A.resolve_t0_date(session=S()) == T0


def test_resolve_t0_falls_back_to_laps_when_session_reference_is_missing():
    """FastF1 sets t0_date to None when telemetry was not loaded or when it
    cannot determine the offset. LapStartDate = LapStartTime + t0_date, so the
    reference can be recovered from the laps table."""
    laps = pd.DataFrame(
        {
            "LapStartTime": pd.to_timedelta([100.0, 190.0, 280.0], unit="s"),
            "LapStartDate": [T0 + pd.Timedelta(seconds=x) for x in (100.0, 190.0, 280.0)],
        }
    )

    class S:
        t0_date = None

    assert A.resolve_t0_date(session=S(), laps=laps) == T0


def test_resolve_t0_returns_none_when_nothing_can_establish_a_reference():
    assert A.resolve_t0_date(session=None, laps=pd.DataFrame()) is None


def test_audit_race_control_converts_absolute_message_timestamps():
    rcm = pd.DataFrame(
        {
            "Time": [T0 + pd.Timedelta(seconds=x) for x in (10.0, 250.0)],
            "Category": ["Flag", "Other"],
            "Message": ["GREEN LIGHT", "VIRTUAL SAFETY CAR DEPLOYED"],
            "Flag": ["GREEN", None],
            "Scope": ["Track", None],
        }
    )
    full, summary = A.audit_race_control(rcm, t0=T0)
    assert full["time_s"].tolist() == pytest.approx([10.0, 250.0])
    assert len(summary) == 2


def test_audit_race_control_writes_the_log_even_without_a_reference():
    """Losing the time axis must not lose the messages. The log is still the
    corroborating record for neutralization timing."""
    rcm = pd.DataFrame(
        {"Time": [T0], "Category": ["Flag"], "Message": ["GREEN LIGHT"],
         "Flag": ["GREEN"], "Scope": ["Track"]}
    )
    full, _ = A.audit_race_control(rcm, t0=None)
    assert len(full) == 1
    assert "time_s" not in full.columns


def test_race_control_and_track_status_share_one_axis():
    """The two neutralization sources must be joinable. If race control were
    converted to epoch seconds this assertion would fail by nine orders of
    magnitude, which is the silent-corruption case the error guards against."""
    rcm = pd.DataFrame(
        {"Time": [T0 + pd.Timedelta(seconds=250.0)], "Category": ["Other"],
         "Message": ["VIRTUAL SAFETY CAR DEPLOYED"], "Flag": [None], "Scope": [None]}
    )
    ts = pd.DataFrame(
        {"Time": pd.to_timedelta([0.0, 250.0], unit="s"), "Status": ["1", "6"],
         "Message": ["AllClear", "VSCDeployed"]}
    )
    rc_s = A.audit_race_control(rcm, t0=T0)[0]["time_s"].iloc[0]
    ts_s = A.audit_track_status(ts).query("status_code == '6'")["time_s"].iloc[0]
    assert rc_s == pytest.approx(ts_s)


def test_audit_session_status_uses_the_shared_converter():
    ss = pd.DataFrame(
        {"Time": pd.to_timedelta([0.0, 500.0], unit="s"),
         "Status": ["Started", "Finished"]}
    )
    out = A.audit_session_status(ss)
    assert out["time_s"].tolist() == pytest.approx([0.0, 500.0])
    assert out.loc[0, "duration_s"] == pytest.approx(500.0)


def test_every_event_table_converter_accepts_a_reference():
    """Guard against a future upstream change to an absolute convention in a
    table that is currently session-relative."""
    import inspect

    for fn in (A.audit_track_status, A.audit_weather, A.audit_race_control,
               A.audit_session_status):
        assert "t0" in inspect.signature(fn).parameters, fn.__name__


# --------------------------------------------------------------------------
# Eligibility chains and pit-event classification
#
# Driven by the first real Phase 0 run, which showed the strict green filter
# discarding 15-18 eligible cars on laps whose only defect was a three-second
# local yellow, and a raw pit-duration column whose mean was 680 s because it
# contained the red-flag suspension.
# --------------------------------------------------------------------------


def _lap_rows(n, status, lap=1.0, compound="INTERMEDIATE"):
    return pd.DataFrame(
        [
            dict(
                Driver=f"D{i}", DriverNumber=str(i), LapNumber=lap, Position=float(i),
                Time=_td(100.0 + 3.0 * i), LapTime=_td(90.0), Compound=compound,
                TrackStatus=status, IsAccurate=True, PitInTime=pd.NaT,
                PitOutTime=pd.NaT, TyreLife=5.0,
            )
            for i in range(1, n + 1)
        ]
    )


def test_relaxed_chain_keeps_yellow_laps_that_the_strict_chain_discards():
    """A brief sector yellow must not disqualify the whole field's lap. On the
    real session this cost laps 4, 36, 40, 41, 42 and 43 their entire sample."""
    laps = _lap_rows(15, status="12")  # green with a yellow segment, no SC/VSC
    gaps = A.compute_same_lap_gaps(laps)
    wet = A.wetness_feasibility(laps, gaps, ["INTERMEDIATE"], ["1"], [1.5]).iloc[0]
    assert wet["n_stage4_green_lap"] == 0
    assert wet["n_stage4_no_neutralization_relaxed"] == 15


def test_relaxed_chain_still_excludes_neutralized_laps():
    """The relaxed filter must not rescue laps that are genuinely unmeasurable.
    Under VSC and SC drivers run to a delta, so lap time carries no grip
    information and the sample should stay empty."""
    for status in ("126", "4", "45", "671"):
        laps = _lap_rows(15, status=status)
        gaps = A.compute_same_lap_gaps(laps)
        wet = A.wetness_feasibility(laps, gaps, ["INTERMEDIATE"], ["1"], [1.5]).iloc[0]
        assert wet["n_stage4_no_neutralization_relaxed"] == 0, status


def test_wetness_observability_flag_respects_the_declared_floor():
    laps = _lap_rows(3, status="1")
    gaps = A.compute_same_lap_gaps(laps)
    below = A.wetness_feasibility(
        laps, gaps, ["INTERMEDIATE"], ["1"], [0.0], min_eligible_cars=4
    ).iloc[0]
    assert bool(below["wetness_observable_relaxed"]) is False

    laps = _lap_rows(6, status="1")
    gaps = A.compute_same_lap_gaps(laps)
    above = A.wetness_feasibility(
        laps, gaps, ["INTERMEDIATE"], ["1"], [0.0], min_eligible_cars=4
    ).iloc[0]
    assert bool(above["wetness_observable_relaxed"]) is True


def test_red_flag_suspension_is_excluded_from_the_duration_model():
    """Suspension durations and stop durations differ by three orders of
    magnitude. Mixing them makes any fitted distribution meaningless."""
    laps = pd.DataFrame(
        [
            dict(Driver="A", DriverNumber="1", LapNumber=32.0, PitInTime=_td(7237.0),
                 PitOutTime=pd.NaT, Compound="INTERMEDIATE", TrackStatus="45",
                 Position=2.0),
            dict(Driver="A", DriverNumber="1", LapNumber=33.0, PitInTime=pd.NaT,
                 PitOutTime=_td(8646.0), Compound="INTERMEDIATE", TrackStatus="51",
                 Position=2.0),
            dict(Driver="B", DriverNumber="2", LapNumber=24.0, PitInTime=_td(6364.0),
                 PitOutTime=pd.NaT, Compound="INTERMEDIATE", TrackStatus="1",
                 Position=6.0),
            dict(Driver="B", DriverNumber="2", LapNumber=25.0, PitInTime=pd.NaT,
                 PitOutTime=_td(6388.6), Compound="INTERMEDIATE", TrackStatus="1",
                 Position=13.0),
        ]
    )
    ts = A.audit_track_status(
        pd.DataFrame(
            {"Time": [_td(0.0), _td(7159.0), _td(8592.0)], "Status": ["1", "5", "1"],
             "Message": ["AllClear", "Red", "AllClear"]}
        )
    )
    ev = A.extract_pit_events(laps, ts).set_index("driver")
    assert bool(ev.loc["A", "is_red_flag_suspension"]) is True
    assert bool(ev.loc["A", "usable_for_duration_model"]) is False
    assert "suspension" in ev.loc["A", "exclusion_reason"]
    assert bool(ev.loc["B", "usable_for_duration_model"]) is True
    assert ev.loc["B", "pit_lane_duration_s"] == pytest.approx(24.6)


def test_retirement_leaves_the_stop_unpaired_and_excluded():
    laps = pd.DataFrame(
        [
            dict(Driver="C", DriverNumber="3", LapNumber=30.0, PitInTime=_td(7020.0),
                 PitOutTime=pd.NaT, Compound="INTERMEDIATE", TrackStatus="4",
                 Position=16.0)
        ]
    )
    ev = A.extract_pit_events(laps, pd.DataFrame()).iloc[0]
    assert bool(ev["is_retirement"]) is True
    assert bool(ev["usable_for_duration_model"]) is False
    assert pd.isna(ev["pit_lane_duration_s"])


def test_all_null_timestamp_column_returns_nan_without_demanding_a_reference():
    """pandas types an all-NaT timedelta column as datetime64. There is nothing
    in it that could be converted wrongly, so it must not raise."""
    s = pd.Series([pd.NaT, pd.NaT])
    out = A.to_session_seconds(s, t0=None, column="PitOutTime")
    assert out.isna().all()
    assert len(out) == 2
