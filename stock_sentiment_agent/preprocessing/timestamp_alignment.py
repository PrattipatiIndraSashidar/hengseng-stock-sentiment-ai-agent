"""Exchange-aware UTC timestamp alignment to prevent look-ahead / future data leakage."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from stock_sentiment_agent.config import EXCHANGE_CONFIGS


def parse_utc_timestamp(ts_str: str) -> datetime:
    """Parse ISO-8601 or common timestamp formats into an aware UTC datetime."""
    if not ts_str:
        return datetime.now(timezone.utc)
    cleaned = ts_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(ts_str.strip(), fmt).replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return datetime.now(timezone.utc)


def is_trading_day(d: date, exchange: str) -> bool:
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    cfg = EXCHANGE_CONFIGS[exchange]
    return d.isoformat() not in cfg.holidays_2026


def next_trading_day(d: date, exchange: str) -> date:
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate, exchange):
        candidate += timedelta(days=1)
    return candidate


def current_or_next_trading_day(d: date, exchange: str) -> date:
    candidate = d
    while not is_trading_day(candidate, exchange):
        candidate += timedelta(days=1)
    return candidate


def align_timestamp_to_trading_session(utc_ts: str | datetime, exchange: str) -> str:
    """Map a UTC event timestamp to the earliest trading session cutoff it can inform.

    Rule (Anti-Leakage):
    - Convert UTC timestamp to the exchange local timezone (HKT or KST).
    - If the event occurred on a non-trading day (weekend/holiday), assign to the next trading day.
    - If the event occurred AFTER the exchange closing bell (16:00 HKT / 15:30 KST), it cannot
      be used to predict that day's close; assign it to the NEXT trading day's feature window.
    """
    cfg = EXCHANGE_CONFIGS[exchange]
    dt_utc = utc_ts if isinstance(utc_ts, datetime) else parse_utc_timestamp(utc_ts)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    local_tz = ZoneInfo(cfg.timezone)
    local_dt = dt_utc.astimezone(local_tz)
    local_date = local_dt.date()

    close_h, close_m = (int(part) for part in cfg.close_time.split(":"))
    cutoff_time = time(close_h, close_m)

    if not is_trading_day(local_date, exchange):
        aligned = current_or_next_trading_day(local_date, exchange)
        return aligned.isoformat()

    if local_dt.time() > cutoff_time:
        aligned = next_trading_day(local_date, exchange)
        return aligned.isoformat()

    return local_date.isoformat()
