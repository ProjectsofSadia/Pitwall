"""Session acquisition from FastF1.

This module is the only place in the project permitted to touch the FastF1
network API. Everything downstream receives dataframes, never a live session
object that could silently re-request data.

Phase 0 note: from Phase 2 onward, modelling and strategy code will not receive
a Session at all. It will receive a TemporalGate constructed from these tables.
That constraint is stated here so the boundary is visible from the first commit.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Must match the pin in pyproject.toml. Two runs of this audit against the same
#: session under FastF1 3.8.1 and 3.8.3 produced different lap-row counts (1135
#: vs 1134) and different sets of drivers flagged all-laps-inaccurate. Version
#: drift is therefore a reproducibility defect, not a detail.
PINNED_FASTF1_VERSION = "3.8.1"


class SessionLoadError(RuntimeError):
    """Raised when a session cannot be loaded or is loaded but unusable."""


class _WarningCollector(logging.Handler):
    """Collects WARNING-and-above records emitted by the FastF1 logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record.getMessage())
        except Exception:  # noqa: BLE001 - a logging handler must never raise
            pass


@dataclass(frozen=True)
class SessionSpec:
    """Identity of the session we intend to load."""

    year: int
    grand_prix: str
    identifier: str

    def __str__(self) -> str:
        return f"{self.year} {self.grand_prix} [{self.identifier}]"


@dataclass
class LoadReport:
    """Provenance record for one session load.

    Serialised into the audit manifest so that any figure or table produced by
    this project can be traced to a specific library version and session.
    """

    requested: dict[str, Any]
    resolved_event_name: str | None = None
    resolved_round: int | None = None
    resolved_location: str | None = None
    resolved_country: str | None = None
    resolved_event_date: str | None = None
    session_name: str | None = None
    session_date_utc: str | None = None
    total_laps: int | None = None
    n_drivers: int | None = None
    fastf1_version: str | None = None
    pandas_version: str | None = None
    numpy_version: str | None = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    loaded_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    cache_dir: str | None = None
    table_row_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: WARNING-or-worse records emitted by FastF1 itself during load. FastF1
    #: swallows several data defects into log warnings rather than exceptions,
    #: so these are part of the audit result, not console noise.
    fastf1_log_warnings: list[str] = field(default_factory=list)
    t0_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_fingerprint(config_path: Path) -> str:
    """SHA-256 of a config file, so a run can be tied to exact settings."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def enable_cache(cache_dir: Path) -> Path:
    """Enable the FastF1 on-disk cache, creating the directory if needed.

    Caching is mandatory for this project, not an optimisation. Without it every
    audit rerun hits the live timing API, which is both slow and a source of
    silent variation if the upstream data is revised.
    """
    import fastf1

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info("FastF1 cache enabled at %s", cache_dir)
    return cache_dir


def load_session(
    spec: SessionSpec,
    cache_dir: Path,
    load_flags: dict[str, bool] | None = None,
):
    """Load one FastF1 session and return ``(session, LoadReport)``.

    Raises
    ------
    SessionLoadError
        If the event cannot be resolved, the download fails, or the session
        loads but the lap table is empty. An empty lap table is treated as a
        failure rather than an empty result because every downstream audit
        would otherwise report zeros that look like findings.
    """
    import fastf1
    import numpy as np
    import pandas as pd

    flags = {"laps": True, "telemetry": True, "weather": True, "messages": True}
    if load_flags:
        flags.update({k: bool(v) for k, v in load_flags.items() if k in flags})

    report = LoadReport(
        requested={"spec": str(spec), **flags},
        fastf1_version=getattr(fastf1, "__version__", "unknown"),
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        cache_dir=str(cache_dir),
    )
    if report.fastf1_version != PINNED_FASTF1_VERSION:
        msg = (
            f"FastF1 {report.fastf1_version} is installed but the project pins "
            f"{PINNED_FASTF1_VERSION}. Lap counts and accuracy flags have been "
            f"observed to differ between patch releases of FastF1, so results "
            f"are not comparable across versions. Recorded in the manifest."
        )
        report.warnings.append(msg)
        logger.warning(msg)

    logger.info(
        "FastF1 %s | pandas %s | numpy %s | python %s",
        report.fastf1_version,
        report.pandas_version,
        report.numpy_version,
        report.python_version,
    )

    try:
        session = fastf1.get_session(spec.year, spec.grand_prix, spec.identifier)
    except Exception as exc:  # noqa: BLE001 - surface the original cause
        raise SessionLoadError(
            f"Could not resolve session {spec}. FastF1 matches the grand prix "
            f"name fuzzily against the event schedule; check config/config.yaml. "
            f"Original error: {exc!r}"
        ) from exc

    # Record what FastF1 actually resolved. Fuzzy matching means the requested
    # string and the resolved event can differ; that must be visible.
    event = getattr(session, "event", None)
    if event is not None:
        report.resolved_event_name = _safe_get(event, "EventName")
        report.resolved_round = _safe_int(_safe_get(event, "RoundNumber"))
        report.resolved_location = _safe_get(event, "Location")
        report.resolved_country = _safe_get(event, "Country")
        report.resolved_event_date = _safe_str(_safe_get(event, "EventDate"))
    report.session_name = getattr(session, "name", None)
    report.session_date_utc = _safe_str(getattr(session, "date", None))
    logger.info(
        "Resolved event: %s (round %s, %s, %s)",
        report.resolved_event_name,
        report.resolved_round,
        report.resolved_location,
        report.resolved_event_date,
    )

    # FastF1 degrades gracefully on partial data failures: its @soft_exceptions
    # decorator logs a warning and continues rather than raising. Those warnings
    # are findings (an all-inaccurate driver removes that car from the wetness
    # sample), so they are captured into the provenance record instead of
    # scrolling past in the console.
    collector = _WarningCollector()
    ff1_logger = logging.getLogger("fastf1")
    ff1_logger.addHandler(collector)
    try:
        session.load(**flags)
    except Exception as exc:  # noqa: BLE001
        raise SessionLoadError(
            f"session.load() failed for {spec} with flags {flags}. If this is a "
            f"network error, retry once; if it persists, the live timing API may "
            f"be unavailable. Original error: {exc!r}"
        ) from exc
    finally:
        ff1_logger.removeHandler(collector)
    report.fastf1_log_warnings = collector.records
    for msg in collector.records:
        logger.info("captured FastF1 warning: %s", msg)

    laps = getattr(session, "laps", None)
    if laps is None or len(laps) == 0:
        raise SessionLoadError(
            f"Session {spec} loaded but session.laps is empty. Refusing to "
            f"continue: downstream audits would report zeros indistinguishable "
            f"from genuine findings."
        )

    report.total_laps = _safe_int(getattr(session, "total_laps", None))
    report.n_drivers = int(laps["Driver"].nunique()) if "Driver" in laps else None
    report.table_row_counts = _table_row_counts(session)
    t0 = getattr(session, "t0_date", None)
    report.t0_date = None if t0 is None else str(t0)

    for table, n in report.table_row_counts.items():
        if n == 0:
            msg = f"Table '{table}' is empty for {spec}."
            report.warnings.append(msg)
            logger.warning(msg)

    logger.info(
        "Loaded %s: %d lap rows, %s drivers, table counts %s",
        spec,
        len(laps),
        report.n_drivers,
        report.table_row_counts,
    )
    return session, report


def _table_row_counts(session) -> dict[str, int]:
    """Row counts for every session table Phase 0 audits."""
    counts: dict[str, int] = {}
    for name in (
        "laps",
        "weather_data",
        "track_status",
        "race_control_messages",
        "session_status",
        "results",
    ):
        try:
            obj = getattr(session, name, None)
            counts[name] = 0 if obj is None else int(len(obj))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read session.%s: %r", name, exc)
            counts[name] = -1
    for name in ("car_data", "pos_data"):
        try:
            obj = getattr(session, name, None)
            counts[name] = 0 if obj is None else int(len(obj))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read session.%s: %r", name, exc)
            counts[name] = -1
    return counts


def _safe_get(obj, key):
    try:
        return obj[key]
    except Exception:  # noqa: BLE001
        return getattr(obj, key, None)


def _safe_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _safe_str(value):
    try:
        if value is None:
            return None
        return str(value)
    except Exception:  # noqa: BLE001
        return None
