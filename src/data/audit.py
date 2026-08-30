"""Phase 0 data audit.

Every function here takes dataframes (or small callables that return them) and
returns dataframes. Nothing in this module performs network IO, which is what
makes the audit logic unit-testable against synthetic frames without a session.

The audit measures. It does not interpret. Interpretation belongs in
docs/DATA_AUDIT.md and is written by a human after reading these outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Approved data classification labels. Nothing else may appear in the registry.
CLASSIFICATIONS: frozenset[str] = frozenset(
    {"OBSERVED", "DERIVED", "ESTIMATED", "ASSUMED", "SIMULATED", "NOT_AVAILABLE"}
)

#: Track status codes as documented in fastf1.api.track_status_data.
#: Code '3' is listed upstream as never observed; retained so an appearance is
#: surfaced rather than silently mapped to "unknown".
TRACK_STATUS_CODES: dict[str, str] = {
    "1": "AllClear",
    "2": "Yellow",
    "3": "Undocumented (upstream reports never observed)",
    "4": "SafetyCar",
    "5": "RedFlag",
    "6": "VSCDeployed",
    "7": "VSCEnding",
}

#: Columns the project design assumes exist on session.laps. Checked, not trusted.
EXPECTED_LAP_COLUMNS: tuple[str, ...] = (
    "Time",
    "Driver",
    "DriverNumber",
    "LapTime",
    "LapNumber",
    "Stint",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Sector1SessionTime",
    "Sector2SessionTime",
    "Sector3SessionTime",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "IsPersonalBest",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Team",
    "LapStartTime",
    "LapStartDate",
    "TrackStatus",
    "Position",
    "Deleted",
    "DeletedReason",
    "FastF1Generated",
    "IsAccurate",
)

EXPECTED_WEATHER_COLUMNS: tuple[str, ...] = (
    "Time",
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindDirection",
    "WindSpeed",
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


class TimestampReferenceError(ValueError):
    """Raised when an absolute timestamp is offered with no session reference.

    Deliberately an error rather than a fallback. Converting an absolute
    datetime to epoch seconds would produce values near 1.73e9 that sort,
    difference and merge without complaint, so every join against ``time_s``
    would be silently wrong by decades. A loud failure is the only safe option.
    """


def resolve_t0_date(session: Any = None, laps: pd.DataFrame | None = None):
    """Determine the timestamp at which session time is zero.

    Preference order:

    1. ``session.t0_date``. This is FastF1's own reference, computed in
       ``_load_telemetry``; the library uses it as
       ``LapStartDate = LapStartTime + t0_date``.
    2. Derived from the laps table as the median of
       ``LapStartDate - LapStartTime``, which is the same identity read
       backwards. Needed because ``t0_date`` is ``None`` when telemetry was not
       loaded, and FastF1 also sets it to ``None`` when it cannot determine the
       offset (it logs "Failed to determine `Session.t0_date`!").
    3. ``None``, meaning absolute timestamps cannot be converted and any attempt
       must raise rather than guess.

    The median in step 2 rather than the first row: individual rows can carry
    NaT, and a single bad row must not define the reference for the session.
    """
    t0 = getattr(session, "t0_date", None) if session is not None else None
    if t0 is not None and not pd.isna(t0):
        return pd.Timestamp(t0)

    if laps is not None and len(laps):
        if "LapStartDate" in laps.columns and "LapStartTime" in laps.columns:
            dates = pd.to_datetime(laps["LapStartDate"], errors="coerce")
            offsets = pd.to_timedelta(laps["LapStartTime"], errors="coerce")
            derived = (dates - offsets).dropna()
            if len(derived):
                t0 = pd.Timestamp(derived.median())
                logger.info(
                    "Session.t0_date unavailable; derived reference %s from "
                    "%d lap rows (LapStartDate - LapStartTime).",
                    t0,
                    len(derived),
                )
                return t0

    logger.warning(
        "No session time reference could be established. Absolute timestamps "
        "cannot be converted to session-relative seconds."
    )
    return None


def _strip_tz(obj):
    """Return a timezone-naive equivalent, or the object unchanged."""
    if isinstance(obj, pd.Series):
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            return obj.dt.tz_convert("UTC").dt.tz_localize(None)
        return obj
    if isinstance(obj, pd.Timestamp) and obj.tzinfo is not None:
        return obj.tz_convert("UTC").tz_localize(None)
    return obj


def to_session_seconds(
    series: pd.Series,
    t0: Any = None,
    column: str = "<unnamed>",
) -> pd.Series:
    """Convert a timestamp column to float seconds since session start.

    FastF1 does not use one timestamp convention across the session tables.
    ``weather_data``, ``track_status`` and ``session_status`` parse their ``Time``
    field with ``to_timedelta``, giving session-relative values. But
    ``race_control_messages`` parses the message ``Utc`` field with
    ``to_datetime``, giving an absolute ``datetime64[ns]``. Both must be handled,
    and they must land in the same units.

    Parameters
    ----------
    series
        Timestamp column, either timedelta64 (already session-relative) or
        datetime64 (absolute, requires ``t0``).
    t0
        Reference timestamp for session time zero, from :func:`resolve_t0_date`.
        Required for datetime64 input, ignored otherwise.
    column
        Column name, used only to make the error message actionable.

    Raises
    ------
    TimestampReferenceError
        If the input is absolute and no reference was supplied, or if the input
        dtype is numeric (a bare number is ambiguous between seconds, epoch
        seconds and nanoseconds, and guessing is how the original defect would
        have become a silent one).
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()

    # An all-null column loses its dtype: pandas types an all-NaT timedelta
    # column as datetime64. There is nothing here that can be converted wrongly,
    # so return NaN rather than demanding a reference for absent data.
    if len(series) and series.isna().all():
        return pd.Series(np.nan, index=series.index, dtype="float64")

    if pd.api.types.is_datetime64_any_dtype(series):
        if t0 is None or pd.isna(t0):
            raise TimestampReferenceError(
                f"Column {column!r} has dtype {series.dtype} (absolute "
                f"timestamps) but no session time reference was supplied. "
                f"Pass t0 from resolve_t0_date(session, laps). Refusing to "
                f"convert: epoch seconds would be wrong by decades and would "
                f"not fail loudly downstream."
            )
        return (_strip_tz(series) - _strip_tz(pd.Timestamp(t0))).dt.total_seconds()

    if pd.api.types.is_numeric_dtype(series):
        raise TimestampReferenceError(
            f"Column {column!r} has numeric dtype {series.dtype}. A bare number "
            f"is ambiguous between session seconds, epoch seconds and "
            f"nanoseconds. Convert it explicitly at the call site."
        )

    converted = pd.to_timedelta(series, errors="coerce")
    if series.notna().any() and converted.isna().all():
        raise TimestampReferenceError(
            f"Column {column!r} with dtype {series.dtype} could not be "
            f"interpreted as a timestamp; every value coerced to NaT."
        )
    return converted.dt.total_seconds()


def _seconds(series: pd.Series, t0: Any = None, column: str = "<unnamed>") -> pd.Series:
    """Backwards-compatible alias for :func:`to_session_seconds`."""
    return to_session_seconds(series, t0=t0, column=column)


def _sample_values(series: pd.Series, n: int = 3) -> str:
    vals = series.dropna().unique()[:n]
    return " | ".join(str(v) for v in vals)


def has_columns(df: pd.DataFrame | None, cols: Iterable[str]) -> bool:
    return df is not None and all(c in df.columns for c in cols)


# --------------------------------------------------------------------------
# Generic table audits
# --------------------------------------------------------------------------


def column_inventory(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Per-column dtype, missingness, cardinality and sample values."""
    rows: list[dict[str, Any]] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        rows.append(
            {
                "table": table_name,
                "column": col,
                "dtype": str(s.dtype),
                "n_rows": n,
                "n_non_null": n - n_null,
                "n_null": n_null,
                "null_frac": round(n_null / n, 6) if n else np.nan,
                "n_unique": int(s.nunique(dropna=True)),
                "sample_values": _sample_values(s),
            }
        )
    return pd.DataFrame(rows)


def expected_column_report(
    df: pd.DataFrame, expected: Sequence[str], table_name: str
) -> pd.DataFrame:
    """Presence check for every column the design assumes exists.

    Absence is a finding, not an exception. A missing column produces a row with
    ``present=False`` so the audit output records exactly what was not there.
    """
    rows: list[dict[str, Any]] = []
    n = len(df)
    for col in expected:
        present = col in df.columns
        n_null = int(df[col].isna().sum()) if present else n
        rows.append(
            {
                "table": table_name,
                "column": col,
                "present": present,
                "dtype": str(df[col].dtype) if present else "",
                "null_frac": round(n_null / n, 6) if n else np.nan,
                "n_unique": int(df[col].nunique(dropna=True)) if present else 0,
            }
        )
    return pd.DataFrame(rows)


def driver_lap_counts(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-driver lap counts and data-quality flags."""
    if "Driver" not in laps.columns:
        return pd.DataFrame()

    def _frac_true(g: pd.DataFrame, col: str) -> float:
        if col not in g.columns or len(g) == 0:
            return np.nan
        return float(pd.Series(g[col]).fillna(False).astype(bool).mean())

    rows: list[dict[str, Any]] = []
    for drv, g in laps.groupby("Driver"):
        rows.append(
            {
                "driver": drv,
                "driver_number": (
                    g["DriverNumber"].iloc[0] if "DriverNumber" in g else ""
                ),
                "team": g["Team"].iloc[0] if "Team" in g else "",
                "n_lap_rows": len(g),
                "max_lap_number": (
                    float(g["LapNumber"].max()) if "LapNumber" in g else np.nan
                ),
                "n_laptime_present": (
                    int(g["LapTime"].notna().sum()) if "LapTime" in g else 0
                ),
                "frac_is_accurate": _frac_true(g, "IsAccurate"),
                "frac_fastf1_generated": _frac_true(g, "FastF1Generated"),
                "frac_deleted": _frac_true(g, "Deleted"),
                "n_pit_in": int(g["PitInTime"].notna().sum()) if "PitInTime" in g else 0,
                "n_pit_out": (
                    int(g["PitOutTime"].notna().sum()) if "PitOutTime" in g else 0
                ),
                "n_stints": (
                    int(g["Stint"].nunique(dropna=True)) if "Stint" in g else 0
                ),
                "compounds_used": (
                    ",".join(sorted(str(c) for c in g["Compound"].dropna().unique()))
                    if "Compound" in g
                    else ""
                ),
                "frac_compound_missing": (
                    float(g["Compound"].isna().mean()) if "Compound" in g else np.nan
                ),
                "frac_tyrelife_missing": (
                    float(g["TyreLife"].isna().mean()) if "TyreLife" in g else np.nan
                ),
                "frac_position_missing": (
                    float(g["Position"].isna().mean()) if "Position" in g else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("driver").reset_index(drop=True)


def per_lap_track_status_distribution(laps: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the concatenated per-lap TrackStatus string.

    FastF1 sets ``laps.TrackStatus`` to the concatenation of every status code
    seen during that lap, so '146' means the lap contained green, safety car and
    VSC-deployed segments. Any eligibility filter that treats this as a single
    code is wrong; this table shows how often mixed laps occur.
    """
    if "TrackStatus" not in laps.columns:
        return pd.DataFrame()
    s = laps["TrackStatus"].astype("string").fillna("")
    counts = s.value_counts(dropna=False).reset_index()
    counts.columns = ["track_status_string", "n_laps"]
    counts["is_pure_green"] = counts["track_status_string"] == "1"
    counts["n_distinct_codes"] = counts["track_status_string"].apply(
        lambda v: len(set(str(v))) if v else 0
    )
    counts["decoded_codes"] = counts["track_status_string"].apply(
        lambda v: ",".join(TRACK_STATUS_CODES.get(c, f"UNKNOWN({c})") for c in str(v))
        if v
        else ""
    )
    counts["frac_laps"] = counts["n_laps"] / max(len(laps), 1)
    return counts


# --------------------------------------------------------------------------
# Track status
# --------------------------------------------------------------------------


def audit_track_status(track_status: pd.DataFrame, t0: Any = None) -> pd.DataFrame:
    """Decode the track status transition log and measure its time resolution.

    ``track_status['Time']`` is parsed by FastF1 with ``to_timedelta`` and is
    session-relative. ``t0`` is accepted so that a future upstream change to an
    absolute convention is handled rather than crashing.
    """
    if track_status is None or len(track_status) == 0:
        return pd.DataFrame()

    df = track_status.copy().reset_index(drop=True)
    if "Time" in df.columns:
        df["time_s"] = to_session_seconds(df["Time"], t0=t0, column="Time")
    else:
        df["time_s"] = np.nan

    if "Status" in df.columns:
        df["status_code"] = df["Status"].astype(str)
        df["decoded"] = df["status_code"].map(
            lambda c: TRACK_STATUS_CODES.get(c, f"UNKNOWN({c})")
        )
    else:
        df["status_code"] = ""
        df["decoded"] = ""

    df = df.sort_values("time_s").reset_index(drop=True)
    df["duration_s"] = df["time_s"].shift(-1) - df["time_s"]
    df["prev_status"] = df["status_code"].shift(1)
    df["transition"] = df["prev_status"].fillna("START") + "->" + df["status_code"]
    return df


def map_time_to_leader_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """Session time at which the race leader started each lap.

    Used only to annotate event tables with an approximate lap number for human
    reading. It is never an input to a model: lap number is a coarse index and
    the project's temporal gate operates on session time.
    """
    if not has_columns(laps, ("LapNumber", "LapStartTime", "Position")):
        return pd.DataFrame(columns=["LapNumber", "leader_lap_start_s"])
    df = laps[["LapNumber", "LapStartTime", "Position"]].copy()
    df["start_s"] = to_session_seconds(df["LapStartTime"], column="LapStartTime")
    df = df.dropna(subset=["LapNumber", "start_s"])
    out = (
        df.groupby("LapNumber")["start_s"]
        .min()
        .reset_index()
        .rename(columns={"start_s": "leader_lap_start_s"})
        .sort_values("LapNumber")
        .reset_index(drop=True)
    )
    return out


def annotate_with_lap(
    df: pd.DataFrame, time_col: str, leader_laps: pd.DataFrame
) -> pd.DataFrame:
    """Attach an approximate lap number to an event table via session time."""
    if df is None or len(df) == 0 or len(leader_laps) == 0:
        return df
    out = df.copy()
    if time_col not in out.columns:
        return out
    edges = leader_laps["leader_lap_start_s"].to_numpy()
    lap_values = leader_laps["LapNumber"].to_numpy()
    idx = np.searchsorted(edges, out[time_col].to_numpy(), side="right") - 1
    idx = np.clip(idx, 0, len(lap_values) - 1)
    approx = lap_values[idx].astype(float)
    approx[out[time_col].isna().to_numpy()] = np.nan
    out["approx_lap"] = approx
    return out


def audit_session_status(
    session_status: pd.DataFrame, t0: Any = None
) -> pd.DataFrame:
    """Session start, suspension, restart and finish transitions.

    Exists so that the conversion goes through :func:`to_session_seconds` rather
    than being reimplemented in the runner script. A second copy of the
    timestamp convention is a second place for it to be wrong.
    """
    if session_status is None or len(session_status) == 0:
        return pd.DataFrame()
    df = session_status.copy().reset_index(drop=True)
    if "Time" in df.columns:
        df["time_s"] = to_session_seconds(
            df["Time"], t0=t0, column="session_status.Time"
        )
        df = df.sort_values("time_s").reset_index(drop=True)
        df["duration_s"] = df["time_s"].shift(-1) - df["time_s"]
    return df


# --------------------------------------------------------------------------
# Race control
# --------------------------------------------------------------------------


def audit_race_control(
    rcm: pd.DataFrame, t0: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(full_messages, category_summary)``.

    Unlike every other session event table, ``race_control_messages['Time']`` is
    absolute ``datetime64[ns]``: FastF1 parses the message ``Utc`` field with
    ``to_datetime`` while parsing weather, track status and session status with
    ``to_timedelta``. A session reference is therefore required to express these
    messages on the same axis as everything else, and ``time_s`` is left absent
    rather than wrong if none is available.

    The full message log is written out unaltered. No message parsing rules are
    applied in Phase 0: deciding which strings mean "VSC deployed" is a
    modelling choice that belongs in a tested parser, not in an audit.
    """
    if rcm is None or len(rcm) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = rcm.copy().reset_index(drop=True)
    if "Time" in df.columns:
        try:
            df["time_s"] = to_session_seconds(df["Time"], t0=t0, column="Time")
        except TimestampReferenceError as exc:
            logger.warning(
                "Race control messages could not be placed on the session time "
                "axis: %s The message log is still written, without time_s.",
                exc,
            )

    group_cols = [c for c in ("Category", "Flag", "Scope") if c in df.columns]
    if group_cols:
        summary = (
            df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="n_messages")
            .sort_values("n_messages", ascending=False)
            .reset_index(drop=True)
        )
    else:
        summary = pd.DataFrame()
    return df, summary


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------


def audit_weather(
    weather: pd.DataFrame, t0: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(channel_summary, sampling_intervals, rainfall_facts)``.

    ``rainfall_facts`` records the measured dtype and unique values of the
    Rainfall channel. The project's decision to replace rainfall with a derived
    wetness index rests on this being a boolean, so it is measured rather than
    assumed.
    """
    if weather is None or len(weather) == 0:
        return pd.DataFrame(), pd.DataFrame(), {"available": False}

    df = weather.copy().reset_index(drop=True)
    if "Time" in df.columns:
        df["time_s"] = to_session_seconds(df["Time"], t0=t0, column="Time")
        df = df.sort_values("time_s").reset_index(drop=True)
        df["dt_s"] = df["time_s"].diff()
    else:
        df["dt_s"] = np.nan

    rows: list[dict[str, Any]] = []
    for col in df.columns:
        if col in ("time_s", "dt_s"):
            continue
        s = df[col]
        row: dict[str, Any] = {
            "channel": col,
            "dtype": str(s.dtype),
            "n_rows": len(s),
            "null_frac": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row.update(
                {
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "median": float(s.median()) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                }
            )
        rows.append(row)
    channel_summary = pd.DataFrame(rows)

    dt = df["dt_s"].dropna()
    sampling = pd.DataFrame(
        [
            {
                "n_samples": len(df),
                "n_intervals": len(dt),
                "dt_min_s": float(dt.min()) if len(dt) else np.nan,
                "dt_p05_s": float(dt.quantile(0.05)) if len(dt) else np.nan,
                "dt_median_s": float(dt.median()) if len(dt) else np.nan,
                "dt_p95_s": float(dt.quantile(0.95)) if len(dt) else np.nan,
                "dt_max_s": float(dt.max()) if len(dt) else np.nan,
                "span_s": (
                    float(df["time_s"].max() - df["time_s"].min())
                    if "time_s" in df
                    else np.nan
                ),
            }
        ]
    )

    facts: dict[str, Any] = {"available": True}
    if "Rainfall" in df.columns:
        s = df["Rainfall"]
        facts.update(
            {
                "rainfall_present": True,
                "rainfall_dtype": str(s.dtype),
                "rainfall_is_bool_dtype": bool(pd.api.types.is_bool_dtype(s)),
                "rainfall_unique_values": sorted(
                    str(v) for v in s.dropna().unique().tolist()
                ),
                "rainfall_n_unique": int(s.nunique(dropna=True)),
                "rainfall_true_frac": (
                    float(s.fillna(False).astype(bool).mean())
                    if s.notna().any()
                    else np.nan
                ),
                "rainfall_intensity_channel_present": False,
            }
        )
    else:
        facts.update({"rainfall_present": False})
    return channel_summary, sampling, facts


# --------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------


def compute_same_lap_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct gaps from line-crossing session times, same lap only.

    Method
    ------
    For a given lap number, every driver's ``Time`` is the session time at which
    that driver crossed the start/finish line completing that lap. Sorting a lap
    group by that time gives the order in which cars crossed, and consecutive
    differences give the gaps between them at that instant.

    This uses only information available at the moment of crossing, so it is
    safe with respect to the no-future-information rule.

    Known invalidity
    ----------------
    The quantity is a gap between cars **on the same lap**. It is not the gap to
    the car physically ahead when that car is a lap up or down. The returned
    ``order_matches_position`` column flags rows where the crossing order and
    the reported classification disagree, which is the observable symptom.
    Under safety car, VSC and around pit cycles the value remains arithmetically
    correct but changes meaning, so consumers must join on track status.
    """
    required = ("Driver", "LapNumber", "Time")
    if not has_columns(laps, required):
        return pd.DataFrame()

    cols = [c for c in ("Driver", "DriverNumber", "LapNumber", "Position", "Time",
                        "TrackStatus", "PitInTime", "PitOutTime", "Compound",
                        "LapTime", "IsAccurate") if c in laps.columns]
    df = laps[cols].copy()
    df["time_s"] = to_session_seconds(df["Time"], column="laps.Time")
    df = df.dropna(subset=["LapNumber", "time_s"])

    frames: list[pd.DataFrame] = []
    for lap, g in df.groupby("LapNumber", sort=True):
        g = g.sort_values("time_s").copy()
        g["crossing_order"] = np.arange(1, len(g) + 1)
        g["gap_ahead_s"] = g["time_s"].diff()
        g["gap_behind_s"] = -g["time_s"].diff(-1)
        g["driver_ahead"] = g["Driver"].shift(1)
        g["driver_behind"] = g["Driver"].shift(-1)
        g["n_cars_on_lap"] = len(g)
        if "Position" in g.columns:
            pos_rank = g["Position"].rank(method="first")
            g["order_matches_position"] = (
                g["crossing_order"].astype(float) == pos_rank.astype(float)
            ).fillna(False)
        else:
            g["order_matches_position"] = False
        frames.append(g)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["LapNumber", "crossing_order"]).reset_index(drop=True)


def gap_diagnostics(gaps: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Per-lap feasibility statistics for the gap reconstruction."""
    if gaps is None or len(gaps) == 0:
        return pd.DataFrame()

    running: dict[float, int] = {}
    if has_columns(laps, ("Driver", "LapNumber")):
        max_lap = laps.groupby("Driver")["LapNumber"].max()
        for lap in sorted(gaps["LapNumber"].unique()):
            running[lap] = int((max_lap >= lap).sum())

    rows: list[dict[str, Any]] = []
    for lap, g in gaps.groupby("LapNumber"):
        n_on_lap = len(g)
        rows.append(
            {
                "LapNumber": float(lap),
                "n_cars_on_lap": n_on_lap,
                "n_cars_still_running": running.get(lap, np.nan),
                "n_gap_ahead_valid": int(g["gap_ahead_s"].notna().sum()),
                "frac_order_matches_position": float(
                    g["order_matches_position"].mean()
                ),
                "median_gap_ahead_s": float(g["gap_ahead_s"].median(skipna=True))
                if g["gap_ahead_s"].notna().any()
                else np.nan,
                "min_gap_ahead_s": float(g["gap_ahead_s"].min(skipna=True))
                if g["gap_ahead_s"].notna().any()
                else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("LapNumber").reset_index(drop=True)
    out["cars_not_on_leader_lap"] = out["n_cars_still_running"] - out["n_cars_on_lap"]
    return out


# --------------------------------------------------------------------------
# Wetness index feasibility
# --------------------------------------------------------------------------


def wetness_feasibility(
    laps: pd.DataFrame,
    gaps: pd.DataFrame,
    wet_compounds: Sequence[str],
    green_status: Sequence[str],
    clean_air_thresholds_s: Sequence[float],
    neutralization_codes: Sequence[str] = ("4", "5", "6", "7"),
    min_eligible_cars: int = 4,
) -> pd.DataFrame:
    """Count the eligible clean-air sample for W(t), lap by lap.

    Filters are applied cumulatively so the output shows exactly which stage
    destroys the sample rather than only the final count. This is the single
    most important Phase 0 measurement: if the eligible field collapses to two
    or three cars during the heavy-rain phase, a field-median wetness index is
    not estimable when it matters most and the methodology must change.

    Two eligibility chains are reported side by side rather than one:

    * **strict** (``green_status``, an exact per-lap ``TrackStatus`` match)
    * **relaxed** (exclude only laps whose status string contains a
      neutralization code)

    Both are emitted because the first Phase 0 run showed the strict chain
    discarding whole laps over a three-second local yellow, while the relaxed
    chain leaves the genuinely unmeasurable laps unmeasurable. Keeping both
    makes the comparison an artifact instead of an argument, and keeps this
    run's columns comparable with the previous one.

    No weighting, scaling or dry-reference correction is applied here. Phase 0
    measures sample availability only.
    """
    if laps is None or len(laps) == 0 or "LapNumber" not in laps.columns:
        return pd.DataFrame()

    gap_lookup: dict[tuple[str, float], float] = {}
    if gaps is not None and len(gaps) and "gap_ahead_s" in gaps.columns:
        for row in gaps.itertuples(index=False):
            gap_lookup[(getattr(row, "Driver"), float(getattr(row, "LapNumber")))] = (
                getattr(row, "gap_ahead_s")
            )

    wet_set = {c.upper() for c in wet_compounds}
    green_set = {str(s) for s in green_status}
    neutral_set = {str(c) for c in neutralization_codes}

    rows: list[dict[str, Any]] = []
    for lap, g in laps.groupby("LapNumber", sort=True):
        rec: dict[str, Any] = {"LapNumber": float(lap), "n_stage0_all_rows": len(g)}

        s1 = g[g["LapTime"].notna()] if "LapTime" in g.columns else g.iloc[0:0]
        rec["n_stage1_laptime"] = len(s1)

        if "IsAccurate" in s1.columns:
            s2 = s1[s1["IsAccurate"].fillna(False).astype(bool)]
        else:
            s2 = s1
        rec["n_stage2_is_accurate"] = len(s2)

        s3 = s2
        if "PitInTime" in s3.columns:
            s3 = s3[s3["PitInTime"].isna()]
        if "PitOutTime" in s3.columns:
            s3 = s3[s3["PitOutTime"].isna()]
        rec["n_stage3_not_pit_lap"] = len(s3)

        def _clean_air_count(frame: pd.DataFrame, thr: float) -> int:
            """Cars in clean air. A car with no car ahead on its lap (the
            leader) has a NaN gap and counts as clean, which is correct: it is
            the single most informative car in the sample."""
            if not len(frame):
                return 0
            gap_vals = np.array(
                [gap_lookup.get((d, float(lap)), np.nan) for d in frame["Driver"]],
                dtype=float,
            )
            return int((np.isnan(gap_vals) | (gap_vals >= float(thr))).sum())

        # --- strict chain: exact per-lap TrackStatus match -------------------
        if "TrackStatus" in s3.columns:
            s4 = s3[s3["TrackStatus"].astype(str).isin(green_set)]
        else:
            s4 = s3
        rec["n_stage4_green_lap"] = len(s4)

        if "Compound" in s4.columns:
            s5 = s4[s4["Compound"].astype(str).str.upper().isin(wet_set)]
        else:
            s5 = s4
        rec["n_stage5_wet_compound"] = len(s5)

        for thr in clean_air_thresholds_s:
            rec[f"n_stage6_clean_air_{thr}s"] = _clean_air_count(s5, thr)

        # --- relaxed chain: exclude neutralization codes only ---------------
        if "TrackStatus" in s3.columns:
            status_str = s3["TrackStatus"].astype(str)
            neutralized = status_str.apply(
                lambda v: any(c in v for c in neutral_set)
            )
            s4r = s3[~neutralized]
        else:
            s4r = s3
        rec["n_stage4_no_neutralization_relaxed"] = len(s4r)

        if "Compound" in s4r.columns:
            s5r = s4r[s4r["Compound"].astype(str).str.upper().isin(wet_set)]
        else:
            s5r = s4r
        rec["n_stage5_wet_compound_relaxed"] = len(s5r)

        for thr in clean_air_thresholds_s:
            rec[f"n_stage6_clean_air_{thr}s_relaxed"] = _clean_air_count(s5r, thr)

        # Whether the relaxed chain yields an observation at all. This is the
        # column that decides where W(t) must be propagated rather than
        # measured, so it is recorded rather than inferred later.
        primary_thr = float(clean_air_thresholds_s[0]) if clean_air_thresholds_s else 0.0
        rec["wetness_observable_relaxed"] = bool(
            rec[f"n_stage6_clean_air_{primary_thr}s_relaxed"] >= min_eligible_cars
        )
        rec["min_eligible_cars"] = int(min_eligible_cars)

        # Observable components of a future W(t), reported without combination.
        for frame, suffix in ((s5, ""), (s5r, "_relaxed")):
            if len(frame) and "LapTime" in frame.columns:
                lt = to_session_seconds(frame["LapTime"], column="LapTime").dropna()
                rec[f"median_laptime_s{suffix}"] = (
                    float(lt.median()) if len(lt) else np.nan
                )
                rec[f"iqr_laptime_s{suffix}"] = (
                    float(lt.quantile(0.75) - lt.quantile(0.25))
                    if len(lt) > 1
                    else np.nan
                )
                rec[f"mad_laptime_s{suffix}"] = (
                    float((lt - lt.median()).abs().median()) if len(lt) else np.nan
                )
            else:
                rec[f"median_laptime_s{suffix}"] = np.nan
                rec[f"iqr_laptime_s{suffix}"] = np.nan
                rec[f"mad_laptime_s{suffix}"] = np.nan

        if "Compound" in g.columns:
            mix = g["Compound"].astype(str).str.upper().value_counts()
            for comp in ("INTERMEDIATE", "WET", "SOFT", "MEDIUM", "HARD"):
                rec[f"n_running_{comp.lower()}"] = int(mix.get(comp, 0))
        rows.append(rec)

    return pd.DataFrame(rows).sort_values("LapNumber").reset_index(drop=True)


def attach_conditions(
    wetness: pd.DataFrame,
    weather: pd.DataFrame,
    leader_laps: pd.DataFrame,
) -> pd.DataFrame:
    """Join the nearest preceding weather sample to each lap.

    Nearest-preceding, never nearest-in-time, so no future observation can be
    attached to a lap. The join is asymmetric on purpose.
    """
    if wetness is None or len(wetness) == 0:
        return wetness
    if weather is None or len(weather) == 0 or len(leader_laps) == 0:
        return wetness
    if "Time" not in weather.columns:
        return wetness

    w = weather.copy()
    w["time_s"] = to_session_seconds(w["Time"], column="weather.Time")
    w = w.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

    out = wetness.merge(leader_laps, on="LapNumber", how="left")
    idx = np.searchsorted(
        w["time_s"].to_numpy(), out["leader_lap_start_s"].to_numpy(), side="right"
    ) - 1
    valid = idx >= 0
    idx_clipped = np.clip(idx, 0, len(w) - 1)
    for col in ("AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"):
        if col in w.columns:
            vals = w[col].to_numpy()[idx_clipped].astype(object)
            vals[~valid] = None
            out[f"weather_{col}"] = vals
    return out


# --------------------------------------------------------------------------
# Pit events
# --------------------------------------------------------------------------


def extract_pit_events(
    laps: pd.DataFrame,
    track_status_audit: pd.DataFrame,
    max_plausible_duration_s: float = 300.0,
) -> pd.DataFrame:
    """Pair pit-in laps with the following pit-out lap and classify the regime.

    The track status at pit entry and at pit exit are recorded separately, along
    with the time to the next status transition. This is the measurement that
    determines whether a stop straddling a neutralization boundary can be
    correctly classified, which the design requires and which lap-level
    resolution cannot deliver.

    Rows are also classified for downstream use. A red-flag suspension is not a
    pit stop: the first Phase 0 run measured green and VSC stops at 24.6-26.6 s
    and suspension "stops" at 1283-1426 s, and a duration distribution fitted
    over the union of those has a mean of 680 s. Retirements leave the in-lap
    unpaired. Both are flagged here rather than left for a modelling script to
    rediscover.
    """
    required = ("Driver", "LapNumber", "PitInTime", "PitOutTime")
    if not has_columns(laps, required):
        return pd.DataFrame()

    df = laps.copy()
    df["pit_in_s"] = to_session_seconds(df["PitInTime"], column="PitInTime")
    df["pit_out_s"] = to_session_seconds(df["PitOutTime"], column="PitOutTime")

    ts_times = np.array([])
    ts_codes: list[str] = []
    if track_status_audit is not None and len(track_status_audit):
        ta = track_status_audit.dropna(subset=["time_s"]).sort_values("time_s")
        ts_times = ta["time_s"].to_numpy()
        ts_codes = ta["status_code"].astype(str).tolist()

    def status_at(t: float) -> tuple[str, float]:
        """Return (status code in force at t, seconds until next transition)."""
        if np.isnan(t) or len(ts_times) == 0:
            return "", np.nan
        i = int(np.searchsorted(ts_times, t, side="right") - 1)
        if i < 0:
            return "", np.nan
        code = ts_codes[i]
        nxt = ts_times[i + 1] - t if i + 1 < len(ts_times) else np.nan
        return code, float(nxt) if nxt == nxt else np.nan

    rows: list[dict[str, Any]] = []
    for drv, g in df.groupby("Driver"):
        g = g.sort_values("LapNumber").reset_index(drop=True)
        for i, row in g.iterrows():
            if pd.isna(row["pit_in_s"]):
                continue
            nxt = g.iloc[i + 1] if i + 1 < len(g) else None
            pit_out_s = (
                float(nxt["pit_out_s"])
                if nxt is not None and not pd.isna(nxt["pit_out_s"])
                else np.nan
            )
            in_code, in_to_next = status_at(float(row["pit_in_s"]))
            out_code, _ = status_at(pit_out_s)
            rows.append(
                {
                    "driver": drv,
                    "driver_number": row.get("DriverNumber", ""),
                    "team": row.get("Team", ""),
                    "in_lap": float(row["LapNumber"]),
                    "out_lap": float(nxt["LapNumber"]) if nxt is not None else np.nan,
                    "pit_in_s": float(row["pit_in_s"]),
                    "pit_out_s": pit_out_s,
                    "pit_lane_duration_s": (
                        pit_out_s - float(row["pit_in_s"])
                        if not np.isnan(pit_out_s)
                        else np.nan
                    ),
                    "compound_before": row.get("Compound", ""),
                    "compound_after": nxt.get("Compound", "") if nxt is not None else "",
                    "tyre_life_before": row.get("TyreLife", np.nan),
                    "tyre_life_after": (
                        nxt.get("TyreLife", np.nan) if nxt is not None else np.nan
                    ),
                    "fresh_tyre_after": (
                        nxt.get("FreshTyre", None) if nxt is not None else None
                    ),
                    "position_before": row.get("Position", np.nan),
                    "position_after": (
                        nxt.get("Position", np.nan) if nxt is not None else np.nan
                    ),
                    "track_status_at_pit_in": in_code,
                    "track_status_at_pit_in_decoded": TRACK_STATUS_CODES.get(
                        in_code, ""
                    ),
                    "track_status_at_pit_out": out_code,
                    "track_status_at_pit_out_decoded": TRACK_STATUS_CODES.get(
                        out_code, ""
                    ),
                    "status_changed_during_stop": bool(
                        in_code and out_code and in_code != out_code
                    ),
                    "seconds_to_next_status_change_from_pit_in": in_to_next,
                    "in_lap_track_status_string": row.get("TrackStatus", ""),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("pit_in_s").reset_index(drop=True)

    out["is_retirement"] = out["pit_out_s"].isna()
    out["is_red_flag_suspension"] = (
        (out["track_status_at_pit_in"].astype(str) == "5")
        | (out["pit_lane_duration_s"] > float(max_plausible_duration_s))
    ).fillna(False)
    out["usable_for_duration_model"] = (
        ~out["is_retirement"] & ~out["is_red_flag_suspension"]
    )
    out["exclusion_reason"] = np.where(
        out["is_retirement"],
        "no out-lap: car retired or did not rejoin",
        np.where(
            out["is_red_flag_suspension"],
            "red-flag suspension, not a pit stop",
            "",
        ),
    )
    return out


def stationary_probe(
    pit_events: pd.DataFrame,
    car_data_getter: Callable[[str], pd.DataFrame | None],
    speed_threshold_kmh: float,
    n_stops: int,
) -> pd.DataFrame:
    """Test whether a stationary segment is detectable in car telemetry.

    Windows the driver's speed trace between pit entry and pit exit and reports
    the longest contiguous run below the speed threshold. Phase 0 asks only
    whether such a run is detectable and plausibly sized; it does not produce a
    stationary-time estimate for modelling.

    FastF1 documents the telemetry ``Time`` column as inaccurate with duplicate
    values and recommends ``Date``. The window is selected on ``Time`` because
    that is the only column commensurate with ``PitInTime``, and durations are
    then measured on ``Date``. That mismatch is itself a finding and is reported
    in the ``window_source`` column.
    """
    if pit_events is None or len(pit_events) == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    subset = pit_events.head(int(n_stops))
    for row in subset.itertuples(index=False):
        num = str(getattr(row, "driver_number", "") or "")
        rec: dict[str, Any] = {
            "driver": getattr(row, "driver", ""),
            "driver_number": num,
            "in_lap": getattr(row, "in_lap", np.nan),
            "pit_lane_duration_s": getattr(row, "pit_lane_duration_s", np.nan),
            "window_source": "car_data.Time; durations from car_data.Date",
        }
        try:
            cd = car_data_getter(num)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)
            rows.append(rec)
            continue
        if cd is None or len(cd) == 0 or "Speed" not in cd.columns:
            rec["error"] = "no car data or no Speed channel"
            rows.append(rec)
            continue

        cd = cd.copy()
        cd["t_s"] = (
            to_session_seconds(cd["Time"], column="car_data.Time")
            if "Time" in cd.columns
            else np.nan
        )
        lo, hi = getattr(row, "pit_in_s", np.nan), getattr(row, "pit_out_s", np.nan)
        if np.isnan(lo) or np.isnan(hi):
            rec["error"] = "incomplete pit window"
            rows.append(rec)
            continue
        win = cd[(cd["t_s"] >= lo) & (cd["t_s"] <= hi)]
        rec["n_samples_in_window"] = len(win)
        if len(win) == 0:
            rec["error"] = "no telemetry samples inside pit window"
            rows.append(rec)
            continue

        below = (win["Speed"] < speed_threshold_kmh).to_numpy()
        rec["n_samples_below_threshold"] = int(below.sum())
        rec["min_speed_kmh"] = float(win["Speed"].min())
        best_len, best_start, cur_len, cur_start = 0, -1, 0, -1
        for i, b in enumerate(below):
            if b:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        rec["longest_below_run_samples"] = int(best_len)
        if best_len > 0 and "Date" in win.columns:
            seg = win.iloc[best_start : best_start + best_len]
            dur = (seg["Date"].iloc[-1] - seg["Date"].iloc[0]).total_seconds()
            rec["longest_below_run_s"] = float(dur)
        else:
            rec["longest_below_run_s"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------


def telemetry_sample_audit(
    drivers: Sequence[str],
    driver_numbers: dict[str, str],
    car_data_getter: Callable[[str], pd.DataFrame | None],
    pos_data_getter: Callable[[str], pd.DataFrame | None],
) -> pd.DataFrame:
    """Channel availability and measured sample rate for a few drivers."""
    rows: list[dict[str, Any]] = []
    for drv in drivers:
        num = driver_numbers.get(drv, "")
        for table_name, getter in (
            ("car_data", car_data_getter),
            ("pos_data", pos_data_getter),
        ):
            rec: dict[str, Any] = {
                "driver": drv,
                "driver_number": num,
                "table": table_name,
            }
            try:
                df = getter(num)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)
                rows.append(rec)
                continue
            if df is None or len(df) == 0:
                rec["error"] = "empty"
                rows.append(rec)
                continue
            rec["n_rows"] = len(df)
            rec["columns"] = ",".join(map(str, df.columns))
            if "Date" in df.columns:
                d = pd.to_datetime(df["Date"]).sort_values()
                dt = d.diff().dt.total_seconds().dropna()
                rec["date_dt_median_ms"] = (
                    float(dt.median() * 1000) if len(dt) else np.nan
                )
                rec["date_dt_p95_ms"] = (
                    float(dt.quantile(0.95) * 1000) if len(dt) else np.nan
                )
                rec["date_min"] = str(d.min())
                rec["date_max"] = str(d.max())
            if "Time" in df.columns:
                t = to_session_seconds(df["Time"], column=f"{table_name}.Time")
                rec["time_min_s"] = float(t.min())
                rec["time_max_s"] = float(t.max())
                rec["time_n_duplicated"] = int(t.duplicated().sum())
            for ch in ("Speed", "RPM", "nGear", "Throttle", "Brake", "DRS",
                       "X", "Y", "Z", "Status"):
                if ch in df.columns:
                    rec[f"null_frac_{ch}"] = float(df[ch].isna().mean())
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Registry resolution
# --------------------------------------------------------------------------


def validate_registry(registry: dict) -> list[str]:
    """Return a list of problems with the variable registry. Empty means valid."""
    problems: list[str] = []
    variables = registry.get("variables") or []
    if not variables:
        problems.append("registry contains no variables")
    seen: set[str] = set()
    for i, var in enumerate(variables):
        name = var.get("name", f"<index {i}>")
        if name in seen:
            problems.append(f"duplicate variable name: {name}")
        seen.add(name)
        cls = var.get("classification")
        if cls not in CLASSIFICATIONS:
            problems.append(
                f"{name}: classification {cls!r} not in {sorted(CLASSIFICATIONS)}"
            )
        if not var.get("resolver"):
            problems.append(f"{name}: missing resolver")
        for key in ("description", "source", "planned_use"):
            if key not in var:
                problems.append(f"{name}: missing {key}")
    return problems


def _leakage_risk(classification: str, resolver: str) -> str:
    """Deterministic leakage-risk label.

    Derived from the classification and resolver rather than hand-written per
    variable, so the rule is auditable in one place:

    * OBSERVED session data is safe only behind the temporal gate.
    * DERIVED quantities inherit that and add estimator-window risk.
    * ESTIMATED from the corpus must exclude this race.
    * SIMULATED and ASSUMED carry no observational leakage.
    * NOT_AVAILABLE carries none.
    """
    if classification == "NOT_AVAILABLE":
        return "none"
    if classification == "OBSERVED":
        return "high if ungated: full-session table, must be read through TemporalGate"
    if classification == "DERIVED":
        return "high: estimator window must be expanding, never full-session"
    if classification == "ESTIMATED":
        return "high: corpus must exclude Sao Paulo 2024"
    if classification == "ASSUMED":
        return "low: fixed a priori, must appear in sensitivity sweep"
    if classification == "SIMULATED":
        return "medium: generator must not condition on post-t observations"
    return "unclassified"


def resolve_registry(registry: dict, context: dict[str, Any]) -> pd.DataFrame:
    """Merge the registry specification with measured availability.

    ``context`` carries what the audit measured:
      laps_columns, weather_columns: {column: null_frac}
      tables: {table_name: row_count}
      car_data_channels, pos_data_channels: set of channel names
      derived: {key: {"available": bool, "note": str, "resolution": str}}
      weather_dt_median_s: float
      telemetry_dt_median_ms: float
    """
    rows: list[dict[str, Any]] = []
    for var in registry.get("variables", []):
        resolver = str(var.get("resolver", ""))
        kind, _, target = resolver.partition(":")
        available: Any = "PENDING"
        resolution = ""
        reliability = ""
        status = "OK"

        if kind == "laps_column":
            cols = context.get("laps_columns", {})
            available = target in cols
            resolution = "per lap (line crossing)"
            if available:
                nf = cols[target]
                reliability = f"null_frac={nf:.4f}"
        elif kind == "weather_column":
            cols = context.get("weather_columns", {})
            available = target in cols
            dt = context.get("weather_dt_median_s")
            resolution = f"~{dt:.1f} s sample interval" if dt else "session sample"
            if available:
                reliability = f"null_frac={cols[target]:.4f}"
        elif kind == "table":
            n = context.get("tables", {}).get(target, 0)
            available = n > 0
            resolution = "event driven (session time)"
            reliability = f"n_rows={n}"
        elif kind == "car_data":
            chans = context.get("car_data_channels", set())
            available = target in chans
            dt = context.get("telemetry_dt_median_ms")
            resolution = f"~{dt:.0f} ms" if dt else "~240 ms (documented)"
        elif kind == "pos_data":
            chans = context.get("pos_data_channels", set())
            available = target in chans
            resolution = "~220 ms (documented)"
        elif kind == "derived":
            info = context.get("derived", {}).get(target, {})
            available = info.get("available", "PENDING")
            resolution = info.get("resolution", "")
            reliability = info.get("note", "")
        elif kind == "absent":
            present_anywhere = (
                var["name"] in context.get("laps_columns", {})
                or var["name"] in context.get("weather_columns", {})
            )
            available = False
            resolution = "n/a"
            reliability = "asserted absent from all loaded tables"
            if present_anywhere:
                status = "MISMATCH: declared NOT_AVAILABLE but a column of that name exists"
        elif kind == "deferred":
            available = "DEFERRED"
            resolution = "introduced in a later phase"
        else:
            status = f"MISMATCH: unknown resolver kind {kind!r}"

        declared = var.get("classification", "")
        if declared == "OBSERVED" and available is False:
            status = "MISMATCH: declared OBSERVED but not present"
        if declared == "NOT_AVAILABLE" and available is True:
            status = "MISMATCH: declared NOT_AVAILABLE but present"

        rows.append(
            {
                "variable": var.get("name"),
                "description": var.get("description"),
                "source": var.get("source"),
                "available": available,
                "temporal_resolution": resolution,
                "classification": declared,
                "reliability": reliability,
                "leakage_risk": _leakage_risk(declared, resolver),
                "planned_use": var.get("planned_use"),
                "notes": var.get("notes", ""),
                "resolver": resolver,
                "status": status,
            }
        )
    return pd.DataFrame(rows)
