"""Historical & live quote market price collector using Yahoo Finance (CookieJar session) + Naver Finance backup.

Priority 3 Compliance:
- Warm-up cookie session via fc.yahoo.com + query2/query1 failover with polite 0.35s spacing.
- 4-hour raw JSON cache in data/raw/prices/ to avoid redundant Yahoo HTTP 429 rate-limits.
- Legitimate secondary market-data provider (Naver Finance siseJson.naver) for KRX equities.
- Never silently substitutes simulated prices when live/historical providers fail.
"""

from __future__ import annotations

import ast
import http.cookiejar
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stock_sentiment_agent.collectors.apify_collector import (
    DATA_CAT_HISTORICAL,
    DATA_CAT_LIVE,
    DATA_CAT_SIMULATED,
)
from stock_sentiment_agent.config import StockConfig, get_settings


@dataclass
class NormalizedPriceBar:
    ticker: str
    trading_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: Optional[float]
    is_simulated: bool = False
    data_category: str = DATA_CAT_HISTORICAL


class PriceCollector:
    """Collects genuine historical OHLCV bars and latest session quote with polite caching."""

    CACHE_TTL_SECONDS = 4 * 3600
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self.sample_path = self.settings.data_dir / "sample_dataset.json"
        self.cache_dir = self.settings.data_dir / "raw" / "prices"
        self._opener: Optional[urllib.request.OpenerDirector] = None
        self._last_request_ts: float = 0.0

    def _get_yahoo_opener(self) -> urllib.request.OpenerDirector:
        if self._opener is not None:
            return self._opener
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        opener.addheaders = [
            ("User-Agent", self.USER_AGENT),
            ("Accept", "application/json,text/plain,*/*"),
        ]
        try:
            opener.open("https://fc.yahoo.com", timeout=5)
        except Exception:
            # fc.yahoo.com sets the session cookie even on HTTP 404
            pass
        self._opener = opener
        return opener

    def _throttle(self, min_interval: float = 0.35) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _read_cache(self, ticker: str) -> Optional[Tuple[List[NormalizedPriceBar], Dict[str, str]]]:
        cache_file = self.cache_dir / f"{ticker.replace('.', '_')}.json"
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(payload["cached_at_utc"])
            age = (datetime.now(timezone.utc) - cached_at).total_seconds()
            if age > self.CACHE_TTL_SECONDS:
                return None
            bars = [NormalizedPriceBar(**b) for b in payload["bars"]]
            meta = payload["meta"]
            meta["reason"] = f"{meta['reason']} (cached {int(age)}s ago)"
            return bars, meta
        except Exception:
            return None

    def _write_cache(self, ticker: str, bars: List[NormalizedPriceBar], meta: Dict[str, str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{ticker.replace('.', '_')}.json"
        cache_file.write_text(
            json.dumps(
                {
                    "cached_at_utc": datetime.now(timezone.utc).isoformat(),
                    "bars": [asdict(b) for b in bars],
                    "meta": meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def collect(
        self,
        stock: StockConfig,
        lookback_days: int = 90,
        allow_simulated_fallback: bool = False,
        use_cache: bool = True,
    ) -> Tuple[List[NormalizedPriceBar], Dict[str, str]]:
        if use_cache:
            cached = self._read_cache(stock.ticker)
            if cached is not None:
                return cached

        # 1. Try yfinance if installed
        try:
            import yfinance as yf  # type: ignore

            df = yf.download(stock.ticker, period=f"{lookback_days}d", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                bars: List[NormalizedPriceBar] = []
                prev_close: Optional[float] = None
                for idx, row in df.iterrows():
                    o = float(row["Open"].iloc[0] if hasattr(row["Open"], "iloc") else row["Open"])
                    h = float(row["High"].iloc[0] if hasattr(row["High"], "iloc") else row["High"])
                    l = float(row["Low"].iloc[0] if hasattr(row["Low"], "iloc") else row["Low"])
                    c = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"])
                    v = float(row["Volume"].iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"])
                    ret = ((c - prev_close) / prev_close) if prev_close and prev_close > 0 else None
                    prev_close = c
                    bars.append(
                        NormalizedPriceBar(
                            ticker=stock.ticker,
                            trading_date=idx.strftime("%Y-%m-%d"),
                            open=round(o, 4),
                            high=round(h, 4),
                            low=round(l, 4),
                            close=round(c, 4),
                            volume=round(v, 2),
                            daily_return=round(ret, 6) if ret is not None else None,
                            is_simulated=False,
                            data_category=DATA_CAT_HISTORICAL,
                        )
                    )
                if bars:
                    meta = {
                        "status": "connected",
                        "provider": "yfinance",
                        "mode": "live_yfinance",
                        "data_category": DATA_CAT_HISTORICAL,
                        "latest_quote_category": DATA_CAT_LIVE,
                        "reason": f"Fetched {len(bars)} genuine OHLCV bars via yfinance.",
                    }
                    self._write_cache(stock.ticker, bars, meta)
                    return bars, meta
        except Exception:
            pass

        # 2. Try Yahoo Finance v8 Chart API with CookieJar + query2/query1 failover + exponential backoff
        err_note = "Empty response"
        opener = self._get_yahoo_opener()
        for attempt, host in enumerate(("query2.finance.yahoo.com", "query1.finance.yahoo.com", "query2.finance.yahoo.com")):
            self._throttle(0.35)
            url = f"https://{host}/v8/finance/chart/{stock.ticker}?range=3mo&interval=1d"
            try:
                with opener.open(url, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                result = data["chart"]["result"][0]
                timestamps = result.get("timestamp", [])
                quote = result["indicators"]["quote"][0]
                bars = []
                prev_close = None
                for i, ts in enumerate(timestamps):
                    o, h, l, c, v = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i], quote["volume"][i]
                    if None in (o, h, l, c, v):
                        continue
                    ret = ((float(c) - prev_close) / prev_close) if prev_close and prev_close > 0 else None
                    prev_close = float(c)
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    bars.append(
                        NormalizedPriceBar(
                            ticker=stock.ticker,
                            trading_date=dt,
                            open=round(float(o), 4),
                            high=round(float(h), 4),
                            low=round(float(l), 4),
                            close=round(float(c), 4),
                            volume=round(float(v), 2),
                            daily_return=round(ret, 6) if ret is not None else None,
                            is_simulated=False,
                            data_category=DATA_CAT_HISTORICAL,
                        )
                    )
                if bars:
                    meta = {
                        "status": "connected",
                        "provider": f"Yahoo Finance ({host})",
                        "mode": "live_yahoo_chart",
                        "data_category": DATA_CAT_HISTORICAL,
                        "latest_quote_category": DATA_CAT_LIVE,
                        "reason": f"Fetched {len(bars)} genuine historical daily bars via {host} (latest session {bars[-1].trading_date}).",
                    }
                    self._write_cache(stock.ticker, bars, meta)
                    return bars, meta
            except urllib.error.HTTPError as exc:
                err_note = f"HTTP {exc.code}: {exc.reason}"
                if exc.code == 429:
                    time.sleep(1.2 * (attempt + 1))
            except Exception as exc:
                err_note = str(exc)

        # 3. Legitimate backup provider for KRX stocks: Naver Finance public chart endpoint
        if stock.exchange == "KRX":
            krx_bars, krx_err = self._collect_naver_krx(stock, lookback_days)
            if krx_bars:
                meta = {
                    "status": "connected",
                    "provider": "Naver Finance Chart API",
                    "mode": "live_naver_chart",
                    "data_category": DATA_CAT_HISTORICAL,
                    "latest_quote_category": DATA_CAT_LIVE,
                    "reason": f"Fetched {len(krx_bars)} genuine KRX bars via Naver Finance fallback (Yahoo error: {err_note}).",
                }
                self._write_cache(stock.ticker, krx_bars, meta)
                return krx_bars, meta
            err_note = f"{err_note}; Naver backup: {krx_err}"

        # 4. Never silently substitute simulated prices unless explicitly opted in
        if allow_simulated_fallback:
            bars = self._load_simulated_prices(stock)
            return bars, {
                "status": "unavailable",
                "provider": "Simulated Demo Dataset",
                "mode": "simulated_demo",
                "data_category": DATA_CAT_SIMULATED,
                "latest_quote_category": DATA_CAT_SIMULATED,
                "reason": f"Market data providers failed ({err_note}); explicit fallback to SIMULATED DEMO DATA.",
            }

        raise RuntimeError(
            f"Genuine market data unavailable for {stock.ticker} ({err_note}). "
            "Refusing to silently substitute simulated prices."
        )

    def _collect_naver_krx(self, stock: StockConfig, lookback_days: int) -> Tuple[List[NormalizedPriceBar], Optional[str]]:
        code = stock.ticker.split(".")[0]
        end_dt = datetime.now(timezone.utc).strftime("%Y%m%d")
        start_dt = (datetime.now(timezone.utc) - timedelta(days=max(lookback_days, 90))).strftime("%Y%m%d")
        url = (
            f"https://api.finance.naver.com/siseJson.naver"
            f"?symbol={code}&requestType=1&startTime={start_dt}&endTime={end_dt}&timeframe=day"
        )
        try:
            self._throttle(0.25)
            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = resp.read().decode("utf-8", errors="replace").strip()
            rows = ast.literal_eval(raw)
            bars: List[NormalizedPriceBar] = []
            prev_close: Optional[float] = None
            for row in rows[1:]:
                if len(row) < 6:
                    continue
                d_raw = str(row[0]).strip()
                t_date = f"{d_raw[:4]}-{d_raw[4:6]}-{d_raw[6:8]}"
                o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
                ret = ((c - prev_close) / prev_close) if prev_close and prev_close > 0 else None
                prev_close = c
                bars.append(
                    NormalizedPriceBar(
                        ticker=stock.ticker,
                        trading_date=t_date,
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=v,
                        daily_return=round(ret, 6) if ret is not None else None,
                        is_simulated=False,
                        data_category=DATA_CAT_HISTORICAL,
                    )
                )
            return bars, None if bars else "No rows parsed from Naver Finance"
        except Exception as exc:
            return [], str(exc)

    def _load_simulated_prices(self, stock: StockConfig) -> List[NormalizedPriceBar]:
        if not self.sample_path.exists():
            return []
        try:
            dataset = json.loads(self.sample_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        bars: List[NormalizedPriceBar] = []
        prev_close: Optional[float] = None
        for row in dataset.get("stock_prices", []):
            if row.get("ticker") != stock.ticker:
                continue
            c = float(row["close"])
            ret = ((c - prev_close) / prev_close) if prev_close and prev_close > 0 else None
            prev_close = c
            bars.append(
                NormalizedPriceBar(
                    ticker=stock.ticker,
                    trading_date=str(row["trading_date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=c,
                    volume=float(row["volume"]),
                    daily_return=round(ret, 6) if ret is not None else None,
                    is_simulated=True,
                    data_category=DATA_CAT_SIMULATED,
                )
            )
        return bars
