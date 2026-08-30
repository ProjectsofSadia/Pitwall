#!/usr/bin/env python3
"""Phase 0 entry point: load the session and write the data audit.

Usage
-----
    python scripts/run_data_audit.py
    python scripts/run_data_audit.py --config config/config.yaml
    python scripts/run_data_audit.py --offline      # cached data only

Writes machine-readable audit tables to ``data/audit/`` and generates
``docs/DATA_AUDIT.md`` from them. The generated document contains measured
tables and explicitly marked NARRATIVE placeholders; the interpretive sections
are written by a human after reading the measurements, not by this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import audit as A  # noqa: E402
from src.data.fastf1_loader import (  # noqa: E402
    SessionLoadError,
    SessionSpec,
    config_fingerprint,
    enable_cache,
    load_session,
)

logger = logging.getLogger("phase0")

NARRATIVE = "<!-- NARRATIVE: written by hand after reviewing the measurements above. -->"


def load_narrative(docs_dir: Path) -> dict[str, str]:
    """Read hand-written interpretation from docs/audit_narrative.md.

    The generated DATA_AUDIT.md interleaves measured tables with interpretation.
    Those two things have different lifecycles: the tables are regenerated on
    every run, the interpretation is written once and revised deliberately.
    Keeping the prose in a separate committed file means rerunning the audit
    cannot silently destroy it.

    Format: level-2 headings whose text matches a section heading in the
    generated document. Everything under a heading becomes that section's
    narrative. Sections with no entry keep the placeholder.
    """
    path = docs_dir / "audit_narrative.md"
    if not path.exists():
        logger.info("No docs/audit_narrative.md; sections will show placeholders.")
        return {}
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    logger.info("Loaded %d narrative sections from %s", len(sections), path.name)
    return sections


# --------------------------------------------------------------------------


def setup_logging(level: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")],
    )


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_csv(df: pd.DataFrame | None, path: Path, label: str) -> int:
    if df is None or len(df) == 0:
        logger.warning("%s: nothing to write (empty result) -> %s", label, path.name)
        pd.DataFrame().to_csv(path, index=False)
        return 0
    df.to_csv(path, index=False)
    logger.info("%s: wrote %d rows -> %s", label, len(df), path.name)
    return len(df)


def df_to_md(df: pd.DataFrame | None, max_rows: int = 40, floatfmt: str = "{:.4g}") -> str:
    """Render a dataframe as a GitHub markdown table without extra dependencies."""
    if df is None or len(df) == 0:
        return "_No rows._"
    shown = df.head(max_rows)

    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    sep = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    out = "\n".join([header, sep, *body])
    if len(df) > max_rows:
        out += f"\n\n_Showing {max_rows} of {len(df)} rows. Full table: see the CSV._"
    return out


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 data audit")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--registry", default="config/variable_registry.yaml")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the FastF1 cache only; fail rather than hitting the network.",
    )
    args = parser.parse_args()

    cfg_path = (REPO_ROOT / args.config).resolve()
    reg_path = (REPO_ROOT / args.registry).resolve()
    cfg = load_config(cfg_path)
    registry = load_config(reg_path)

    audit_dir = (REPO_ROOT / cfg["paths"]["audit_dir"]).resolve()
    docs_dir = (REPO_ROOT / cfg["paths"]["docs_dir"]).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"), audit_dir / "audit_run.log")
    logger.info("Repository root: %s", REPO_ROOT)

    problems = A.validate_registry(registry)
    if problems:
        for p in problems:
            logger.error("registry problem: %s", p)
        logger.error("Refusing to run with an invalid variable registry.")
        return 2
    logger.info("Variable registry valid: %d variables", len(registry["variables"]))

    # ---------------------------------------------------------------- load
    cache_dir = enable_cache(REPO_ROOT / cfg["paths"]["cache_dir"])
    if args.offline:
        import fastf1

        fastf1.Cache.offline_mode(True)
        logger.info("FastF1 offline mode enabled: cache only.")

    spec = SessionSpec(
        year=int(cfg["session"]["year"]),
        grand_prix=str(cfg["session"]["grand_prix"]),
        identifier=str(cfg["session"]["identifier"]),
    )
    try:
        session, report = load_session(spec, cache_dir, cfg["session"].get("load"))
    except SessionLoadError as exc:
        logger.error("PHASE 0 ABORTED: %s", exc)
        return 3

    return run_audit(
        session=session,
        report=report,
        cfg=cfg,
        registry=registry,
        audit_dir=audit_dir,
        docs_dir=docs_dir,
        cfg_path=cfg_path,
        reg_path=reg_path,
        offline=bool(args.offline),
    )


def run_audit(
    session,
    report,
    cfg: dict,
    registry: dict,
    audit_dir: Path,
    docs_dir: Path,
    cfg_path: Path,
    reg_path: Path,
    offline: bool = False,
) -> int:
    """Audit an already-loaded session and write all Phase 0 outputs.

    Separated from ``main`` so the full pipeline can be exercised offline
    against a synthetic session in the test suite. Everything below this point
    is pure dataframe work plus file writes.
    """
    laps = pd.DataFrame(session.laps)
    weather = pd.DataFrame(getattr(session, "weather_data", pd.DataFrame()))
    track_status = pd.DataFrame(getattr(session, "track_status", pd.DataFrame()))
    rcm = pd.DataFrame(getattr(session, "race_control_messages", pd.DataFrame()))
    session_status = pd.DataFrame(getattr(session, "session_status", pd.DataFrame()))
    results = pd.DataFrame(getattr(session, "results", pd.DataFrame()))

    # Establish the session time reference before any timestamp conversion.
    # race_control_messages['Time'] is absolute datetime64 while every other
    # event table is session-relative timedelta64, so a reference is required
    # to place them on a common axis.
    t0 = A.resolve_t0_date(session=session, laps=laps)
    logger.info("Session time reference (t0_date): %s", t0)
    if t0 is None:
        logger.warning(
            "No session time reference. Race control messages will be written "
            "without a session-relative time_s column."
        )

    written: dict[str, int] = {}
    acfg = cfg["audit"]

    # ------------------------------------------------- inventories
    inventories = [
        A.column_inventory(laps, "laps"),
        A.column_inventory(weather, "weather_data"),
        A.column_inventory(track_status, "track_status"),
        A.column_inventory(rcm, "race_control_messages"),
        A.column_inventory(session_status, "session_status"),
        A.column_inventory(results, "results"),
    ]
    inventory = pd.concat([d for d in inventories if len(d)], ignore_index=True)
    written["session_columns"] = write_csv(
        inventory, audit_dir / "session_columns.csv", "column inventory"
    )
    written["missingness"] = write_csv(
        inventory[["table", "column", "n_rows", "n_null", "null_frac"]],
        audit_dir / "missingness.csv",
        "missingness",
    )

    expected = pd.concat(
        [
            A.expected_column_report(laps, A.EXPECTED_LAP_COLUMNS, "laps"),
            A.expected_column_report(weather, A.EXPECTED_WEATHER_COLUMNS, "weather_data"),
        ],
        ignore_index=True,
    )
    written["expected_columns"] = write_csv(
        expected, audit_dir / "expected_columns.csv", "expected columns"
    )
    missing_expected = expected[~expected["present"]]
    if len(missing_expected):
        logger.warning(
            "MISSING EXPECTED COLUMNS: %s",
            ", ".join(f"{r.table}.{r.column}" for r in missing_expected.itertuples()),
        )

    dlc = A.driver_lap_counts(laps)
    written["driver_lap_counts"] = write_csv(
        dlc, audit_dir / "driver_lap_counts.csv", "driver lap counts"
    )

    lts = A.per_lap_track_status_distribution(laps)
    written["lap_track_status_distribution"] = write_csv(
        lts, audit_dir / "lap_track_status_distribution.csv", "per-lap track status"
    )

    # ------------------------------------------------- track status
    leader_laps = A.map_time_to_leader_lap(laps)
    ts_audit = A.audit_track_status(track_status, t0=t0)
    ts_audit = A.annotate_with_lap(ts_audit, "time_s", leader_laps)
    written["track_status"] = write_csv(
        ts_audit, audit_dir / "track_status.csv", "track status transitions"
    )

    ss = A.audit_session_status(session_status, t0=t0)
    if len(ss) and "time_s" in ss.columns:
        ss = A.annotate_with_lap(ss, "time_s", leader_laps)
    written["session_status"] = write_csv(
        ss, audit_dir / "session_status.csv", "session status"
    )

    # ------------------------------------------------- race control
    rcm_full, rcm_summary = A.audit_race_control(rcm, t0=t0)
    if len(rcm_full) and "time_s" in rcm_full.columns:
        rcm_full = A.annotate_with_lap(rcm_full, "time_s", leader_laps)
    written["race_control_messages"] = write_csv(
        rcm_full, audit_dir / "race_control_messages.csv", "race control messages"
    )
    written["race_control_summary"] = write_csv(
        rcm_summary, audit_dir / "race_control_summary.csv", "race control summary"
    )

    # ------------------------------------------------- weather
    w_channels, w_sampling, rainfall_facts = A.audit_weather(weather, t0=t0)
    written["weather_audit"] = write_csv(
        w_channels, audit_dir / "weather_audit.csv", "weather channels"
    )
    written["weather_sampling"] = write_csv(
        w_sampling, audit_dir / "weather_sampling.csv", "weather sampling"
    )
    logger.info("Rainfall facts: %s", json.dumps(rainfall_facts, default=str))

    # ------------------------------------------------- gaps
    gaps = A.compute_same_lap_gaps(laps)
    written["gaps"] = write_csv(gaps, audit_dir / "gaps.csv", "gap reconstruction")
    gap_diag = A.gap_diagnostics(gaps, laps)
    written["gap_diagnostics"] = write_csv(
        gap_diag, audit_dir / "gap_diagnostics.csv", "gap diagnostics"
    )

    # ------------------------------------------------- wetness feasibility
    wet = A.wetness_feasibility(
        laps,
        gaps,
        wet_compounds=acfg["wet_compounds"],
        green_status=acfg["green_lap_track_status"],
        clean_air_thresholds_s=acfg["clean_air_gap_thresholds_s"],
        neutralization_codes=acfg.get(
            "neutralization_track_status_codes", ["4", "5", "6", "7"]
        ),
        min_eligible_cars=int(acfg.get("min_eligible_cars_for_wetness", 4)),
    )
    wet = A.attach_conditions(wet, weather, leader_laps)
    written["wetness_feasibility"] = write_csv(
        wet, audit_dir / "wetness_feasibility.csv", "wetness feasibility"
    )

    # ------------------------------------------------- pit events
    pit_events = A.extract_pit_events(
        laps,
        ts_audit,
        max_plausible_duration_s=float(
            acfg.get("max_plausible_pit_lane_duration_s", 300.0)
        ),
    )
    written["pit_events"] = write_csv(
        pit_events, audit_dir / "pit_events.csv", "pit events"
    )

    car_data = getattr(session, "car_data", {}) or {}
    pos_data = getattr(session, "pos_data", {}) or {}

    def car_get(num: str):
        return pd.DataFrame(car_data[num]) if num in car_data else None

    def pos_get(num: str):
        return pd.DataFrame(pos_data[num]) if num in pos_data else None

    # Probe only stops that are actually stops. Windowing telemetry across a
    # 23-minute red-flag suspension would measure the grid, not a pit box.
    probe_source = (
        pit_events[pit_events["usable_for_duration_model"]]
        if len(pit_events) and "usable_for_duration_model" in pit_events.columns
        else pit_events
    )
    probe = A.stationary_probe(
        probe_source,
        car_get,
        speed_threshold_kmh=float(acfg["stationary_speed_kmh"]),
        n_stops=int(acfg["stationary_probe_n_stops"]),
    )
    written["stationary_probe"] = write_csv(
        probe, audit_dir / "stationary_probe.csv", "stationary probe"
    )

    # ------------------------------------------------- telemetry
    numbers: dict[str, str] = {}
    if "DriverNumber" in laps.columns:
        numbers = (
            laps.dropna(subset=["Driver"])
            .groupby("Driver")["DriverNumber"]
            .first()
            .astype(str)
            .to_dict()
        )
    sample_drivers = [d for d in acfg["telemetry_sample_drivers"] if d in numbers]
    if len(sample_drivers) < len(acfg["telemetry_sample_drivers"]):
        logger.warning(
            "Telemetry sample drivers not all present. Requested %s, using %s",
            acfg["telemetry_sample_drivers"],
            sample_drivers,
        )
    tel = A.telemetry_sample_audit(sample_drivers, numbers, car_get, pos_get)
    written["telemetry_audit"] = write_csv(
        tel, audit_dir / "telemetry_audit.csv", "telemetry audit"
    )

    # ------------------------------------------------- registry resolution
    def null_map(df: pd.DataFrame) -> dict[str, float]:
        return {c: float(df[c].isna().mean()) for c in df.columns} if len(df) else {}

    car_channels: set[str] = set()
    pos_channels: set[str] = set()
    if sample_drivers:
        first = numbers[sample_drivers[0]]
        cd, pdta = car_get(first), pos_get(first)
        car_channels = set(cd.columns) if cd is not None else set()
        pos_channels = set(pdta.columns) if pdta is not None else set()

    tel_dt = np.nan
    if len(tel) and "date_dt_median_ms" in tel.columns:
        car_rows = tel[tel["table"] == "car_data"]
        if len(car_rows):
            tel_dt = float(car_rows["date_dt_median_ms"].median())

    def derived_info(df: pd.DataFrame, key: str, resolution: str) -> dict[str, Any]:
        return {
            "available": bool(df is not None and len(df) > 0),
            "note": f"{key}: {0 if df is None else len(df)} rows produced",
            "resolution": resolution,
        }

    context = {
        "laps_columns": null_map(laps),
        "weather_columns": null_map(weather),
        "tables": {
            "laps": len(laps),
            "weather_data": len(weather),
            "track_status": len(track_status),
            "race_control_messages": len(rcm),
            "session_status": len(session_status),
            "results": len(results),
        },
        "car_data_channels": car_channels,
        "pos_data_channels": pos_channels,
        "weather_dt_median_s": (
            float(w_sampling["dt_median_s"].iloc[0]) if len(w_sampling) else None
        ),
        "telemetry_dt_median_ms": None if np.isnan(tel_dt) else tel_dt,
        "derived": {
            "gaps": derived_info(gaps, "gaps", "per lap (line crossing)"),
            "wetness_feasibility": derived_info(
                wet, "wetness_feasibility", "per lap"
            ),
            "pit_events": derived_info(pit_events, "pit_events", "per stop, session time"),
            "stationary_probe": derived_info(
                probe, "stationary_probe", "telemetry sample rate"
            ),
        },
    }
    classification = A.resolve_registry(registry, context)
    written["data_classification"] = write_csv(
        classification, audit_dir / "data_classification.csv", "data classification"
    )
    mismatches = classification[classification["status"] != "OK"]
    for row in mismatches.itertuples(index=False):
        logger.warning("REGISTRY %s -> %s", row.variable, row.status)

    # ------------------------------------------------- manifest
    manifest = {
        "load_report": report.to_dict(),
        "config_path": _rel(cfg_path),
        "config_sha256": config_fingerprint(cfg_path),
        "registry_path": _rel(reg_path),
        "registry_sha256": config_fingerprint(reg_path),
        "offline_mode": bool(offline),
        "rows_written": written,
        "rainfall_facts": rainfall_facts,
        "session_time_reference_t0_date": str(t0) if t0 is not None else None,
        "race_control_time_axis_resolved": bool(
            len(rcm_full) and "time_s" in rcm_full.columns
        ),
        "drivers_with_zero_accurate_laps": (
            sorted(dlc.loc[dlc["frac_is_accurate"] == 0.0, "driver"].tolist())
            if len(dlc) and "frac_is_accurate" in dlc.columns
            else []
        ),
        "n_drivers_in_laps": int(laps["Driver"].nunique()) if len(laps) else 0,
        "n_drivers_in_results": int(len(results)),
        "pos_data_keys_not_in_laps": sorted(
            set(map(str, (getattr(session, "pos_data", {}) or {}).keys()))
            - set(laps["DriverNumber"].astype(str).unique())
        )
        if len(laps) and "DriverNumber" in laps.columns
        else [],
        "n_pit_stops_total": int(len(pit_events)),
        "n_pit_stops_usable_for_duration_model": (
            int(pit_events["usable_for_duration_model"].sum())
            if len(pit_events) and "usable_for_duration_model" in pit_events.columns
            else 0
        ),
        "n_pit_stops_straddling_status_change": (
            int(pit_events["status_changed_during_stop"].sum())
            if len(pit_events) and "status_changed_during_stop" in pit_events.columns
            else 0
        ),
        "n_laps_wetness_observable_relaxed": (
            int(wet["wetness_observable_relaxed"].sum())
            if len(wet) and "wetness_observable_relaxed" in wet.columns
            else 0
        ),
        "laps_wetness_not_observable_relaxed": (
            sorted(
                wet.loc[~wet["wetness_observable_relaxed"], "LapNumber"]
                .astype(int)
                .tolist()
            )
            if len(wet) and "wetness_observable_relaxed" in wet.columns
            else []
        ),
        "n_registry_mismatches": int(len(mismatches)),
        "missing_expected_columns": [
            f"{r.table}.{r.column}" for r in missing_expected.itertuples()
        ],
    }
    (audit_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote run_manifest.json")

    # ------------------------------------------------- document
    doc = build_document(
        narrative=load_narrative(docs_dir),
        report=report,
        manifest=manifest,
        laps=laps,
        expected=expected,
        dlc=dlc,
        lts=lts,
        ts_audit=ts_audit,
        session_status=ss,
        rcm_summary=rcm_summary,
        w_channels=w_channels,
        w_sampling=w_sampling,
        rainfall_facts=rainfall_facts,
        tel=tel,
        gap_diag=gap_diag,
        wet=wet,
        pit_events=pit_events,
        probe=probe,
        classification=classification,
    )
    (docs_dir / "DATA_AUDIT.md").write_text(doc, encoding="utf-8")
    logger.info("Wrote docs/DATA_AUDIT.md")

    logger.info("PHASE 0 AUDIT COMPLETE. %d registry mismatches.", len(mismatches))
    return 0


# --------------------------------------------------------------------------


def build_document(**k: Any) -> str:
    """Assemble DATA_AUDIT.md from measured tables plus narrative placeholders."""
    r = k["report"]
    rf = k["rainfall_facts"]
    narrative: dict[str, str] = k.get("narrative") or {}
    parts: list[str] = []
    add = parts.append

    def narr(section: str) -> str:
        """Hand-written interpretation for a section, or the placeholder."""
        return narrative.get(section, NARRATIVE)

    add("# Data Audit - 2024 Sao Paulo Grand Prix\n")
    add(
        "Generated by `scripts/run_data_audit.py`. Every table below is measured "
        "from the loaded session. Sections marked NARRATIVE are written by hand "
        "after reading the measurements and are not produced by the script.\n"
    )

    add("## Purpose\n")
    add(
        "Establish, against the loaded session rather than against documentation, "
        "which variables the Interlagos Decision Lab methodology depends on are "
        "actually obtainable, at what temporal resolution, and with what "
        "reliability. Phase 0 has veto power over the Phase 1 design.\n"
    )

    add("## FastF1 Environment\n")
    add(
        f"- FastF1 `{r.fastf1_version}`, pandas `{r.pandas_version}`, "
        f"numpy `{r.numpy_version}`, Python `{r.python_version}`\n"
        f"- Platform: `{r.platform}`\n"
        f"- Loaded at (UTC): `{r.loaded_at_utc}`\n"
        f"- Cache: `{r.cache_dir}`\n"
        f"- Session time reference (t0_date): "
        f"`{k['manifest'].get('session_time_reference_t0_date')}`\n"
        f"- Config SHA-256: `{k['manifest']['config_sha256']}`\n"
        f"- Registry SHA-256: `{k['manifest']['registry_sha256']}`\n"
    )

    add("## Session Overview\n")
    add(
        f"- Requested: `{r.requested}`\n"
        f"- Resolved event: **{r.resolved_event_name}** "
        f"(round {r.resolved_round}, {r.resolved_location}, {r.resolved_country})\n"
        f"- Event date: `{r.resolved_event_date}` | Session: `{r.session_name}` "
        f"at `{r.session_date_utc}`\n"
        f"- Scheduled total laps: `{r.total_laps}` | Drivers with lap data: "
        f"`{r.n_drivers}`\n"
        f"- Table row counts: `{r.table_row_counts}`\n"
    )
    if r.warnings:
        add("Loader warnings:\n")
        for w in r.warnings:
            add(f"- {w}")
        add("")
    if getattr(r, "fastf1_log_warnings", None):
        add(
            "Warnings emitted by FastF1 itself during load. FastF1 degrades "
            "gracefully on partial data failures rather than raising, so these "
            "are audit findings, not console noise:\n"
        )
        for w in r.fastf1_log_warnings:
            add(f"- `{w}`")
        add("")
    zero_acc = k["manifest"].get("drivers_with_zero_accurate_laps") or []
    add(
        f"- Drivers with no accurate laps at all: `{zero_acc if zero_acc else 'none'}`\n"
        f"- Drivers in laps: `{k['manifest'].get('n_drivers_in_laps')}` | "
        f"in results: `{k['manifest'].get('n_drivers_in_results')}`\n"
        f"- Position-data keys not matching any lap driver number: "
        f"`{k['manifest'].get('pos_data_keys_not_in_laps')}`\n"
    )

    add("## Lap Timing Data\n")
    add("Presence and missingness of every column the design assumes exists:\n")
    add(df_to_md(k["expected"][k["expected"]["table"] == "laps"], max_rows=40) + "\n")
    add("Per-driver lap counts and quality flags:\n")
    add(df_to_md(k["dlc"], max_rows=25) + "\n")
    add(
        "Distribution of the concatenated per-lap `TrackStatus` string. A value "
        "such as `146` means the lap contained green, safety car and "
        "VSC-deployed segments; any filter treating this as a single code is "
        "wrong:\n"
    )
    add(df_to_md(k["lts"], max_rows=25) + "\n")
    add(narr("Lap Timing Data") + "\n")

    add("## Track Status\n")
    add("Full transition log with decoded codes and time to the next transition:\n")
    add(df_to_md(k["ts_audit"], max_rows=60) + "\n")
    add("Session status transitions (start, suspension, restart, finish):\n")
    add(df_to_md(k["session_status"], max_rows=30) + "\n")
    add(narr("Track Status") + "\n")

    add("## Race Control\n")
    add("Message counts by category, flag and scope:\n")
    add(df_to_md(k["rcm_summary"], max_rows=40) + "\n")
    add(
        "The full message log is in `data/audit/race_control_messages.csv`. No "
        "parsing rules were applied in Phase 0: deciding which strings denote a "
        "VSC deployment is a modelling choice that belongs in a tested parser.\n"
    )
    add(narr("Race Control") + "\n")

    add("## Weather\n")
    add(df_to_md(k["w_channels"], max_rows=20) + "\n")
    add("Measured sampling interval:\n")
    add(df_to_md(k["w_sampling"], max_rows=5) + "\n")
    add("Rainfall channel, measured rather than assumed:\n")
    add("```json\n" + json.dumps(rf, indent=2, default=str) + "\n```\n")
    if rf.get("rainfall_present") and rf.get("rainfall_is_bool_dtype"):
        add(
            "**Rainfall intensity is not available from FastF1 weather data.** The "
            "channel is boolean. No mm/h value is inferred anywhere in this "
            "project. The weather state is the derived field wetness index "
            "W(t) instead.\n"
        )
    add(narr("Weather") + "\n")

    add("## Telemetry\n")
    add(df_to_md(k["tel"][k["tel"]["table"] == "car_data"], max_rows=10) + "\n")
    add(narr("Telemetry") + "\n")

    add("## Position Data\n")
    add(df_to_md(k["tel"][k["tel"]["table"] == "pos_data"], max_rows=10) + "\n")
    add(narr("Position Data") + "\n")

    add("## Pit Stop Observability\n")
    add(
        "Every stop, with the track status in force at pit entry and at pit exit "
        "and the time to the next status transition. A stop where "
        "`status_changed_during_stop` is true straddles a neutralization "
        "boundary and cannot be assigned a single pit-loss regime:\n"
    )
    add(df_to_md(k["pit_events"], max_rows=60) + "\n")
    add("Stationary-segment detectability probe on a sample of stops:\n")
    add(df_to_md(k["probe"], max_rows=15) + "\n")
    add(narr("Pit Stop Observability") + "\n")

    add("## Gap Reconstruction\n")
    add(
        "Gaps are reconstructed from start/finish line crossing session times "
        "within a lap group. This uses only information available at the moment "
        "of crossing. It is a same-lap quantity and is not the gap to a car that "
        "is a lap up or down; `frac_order_matches_position` and "
        "`cars_not_on_leader_lap` measure how often that distinction bites:\n"
    )
    add(df_to_md(k["gap_diag"], max_rows=80) + "\n")
    add(narr("Gap Reconstruction") + "\n")

    add("## Wetness Index Feasibility\n")
    add(
        "Cumulative eligibility filter, lap by lap. Stage 0 is all lap rows; each "
        "subsequent stage adds one condition. The question this table answers is "
        "whether the clean-air sample survives the heavy-rain phase:\n"
    )
    add(df_to_md(k["wet"], max_rows=80) + "\n")
    add(narr("Wetness Index Feasibility") + "\n")

    add("## Temporal Resolution\n")
    add(narr("Temporal Resolution") + "\n")

    add("## Missingness and Reliability\n")
    add(
        "Full per-column missingness for every audited table is in "
        "`data/audit/missingness.csv`.\n"
    )
    missing = k["manifest"]["missing_expected_columns"]
    add(
        f"Expected columns absent from the loaded session: "
        f"`{missing if missing else 'none'}`\n"
    )
    add(narr("Missingness and Reliability") + "\n")

    add("## Data Classification\n")
    add(
        "Specification from `config/variable_registry.yaml` resolved against the "
        "loaded session. `status` is `MISMATCH` wherever the declared "
        "classification disagrees with what was measured:\n"
    )
    add(df_to_md(k["classification"], max_rows=60) + "\n")
    add(narr("Data Classification") + "\n")

    add("## Known Limitations\n")
    add(narr("Known Limitations") + "\n")

    add("## Implications for Methodology\n")
    add(narr("Implications for Methodology") + "\n")

    add("## Phase 0 Verdict\n")
    add(
        "Each proposed component is classified FEASIBLE, FEASIBLE WITH "
        "ESTIMATION, FEASIBLE WITH STRONG ASSUMPTIONS, or NOT DEFENSIBLE.\n"
    )
    add(narr("Phase 0 Verdict") + "\n")

    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
