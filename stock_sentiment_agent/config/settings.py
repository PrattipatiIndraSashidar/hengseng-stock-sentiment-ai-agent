"""Centralized configuration, stock universe, exchange calendars, and secure env validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent


def _load_env_files() -> None:
    """Load .env files via python-dotenv or stdlib parser without overriding existing env vars."""
    candidates = [WORKSPACE_ROOT / ".env", BASE_DIR / ".env"]
    try:
        from dotenv import load_dotenv  # type: ignore

        for env_path in candidates:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            continue


_load_env_files()


@dataclass(frozen=True)
class ExchangeConfig:
    code: str
    name: str
    timezone: str
    currency: str
    currency_symbol: str
    open_time: str
    close_time: str
    social_sources: List[str]
    holidays_2026: List[str] = field(default_factory=list)


@dataclass
class StockConfig:
    ticker: str
    company_name: str
    exchange: str
    sector: str
    aliases: List[str] = field(default_factory=list)
    enabled: bool = True


EXCHANGE_CONFIGS: Dict[str, ExchangeConfig] = {
    "HKEX": ExchangeConfig(
        code="HKEX",
        name="Hong Kong Stock Exchange",
        timezone="Asia/Hong_Kong",
        currency="HKD",
        currency_symbol="HK$",
        open_time="09:30",
        close_time="16:00",
        social_sources=["LIHKG", "Telegram"],
        holidays_2026=[
            "2026-01-01", "2026-02-17", "2026-02-18", "2026-02-19",
            "2026-04-03", "2026-04-06", "2026-04-07", "2026-05-01",
            "2026-05-25", "2026-06-19", "2026-07-01", "2026-09-26",
            "2026-10-01", "2026-10-19", "2026-12-25", "2026-12-26",
        ],
    ),
    "KRX": ExchangeConfig(
        code="KRX",
        name="Korea Exchange",
        timezone="Asia/Seoul",
        currency="KRW",
        currency_symbol="₩",
        open_time="09:00",
        close_time="15:30",
        social_sources=["Naver Finance", "Daum Finance"],
        holidays_2026=[
            "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18",
            "2026-03-01", "2026-05-01", "2026-05-05", "2026-05-24",
            "2026-06-06", "2026-08-15", "2026-09-24", "2026-09-25",
            "2026-09-26", "2026-10-03", "2026-10-09", "2026-12-25",
            "2026-12-31",
        ],
    ),
}


DEFAULT_STOCKS: List[StockConfig] = [
    StockConfig("0700.HK", "Tencent", "HKEX", "Communication Services", ["騰訊", "Tencent Holdings", "0700", "700.HK"]),
    StockConfig("9988.HK", "Alibaba", "HKEX", "Consumer Cyclical", ["阿里巴巴", "Alibaba Group", "9988", "BABA"]),
    StockConfig("3690.HK", "Meituan", "HKEX", "Consumer Cyclical", ["美團", "Meituan Dianping", "3690"]),
    StockConfig("1810.HK", "Xiaomi", "HKEX", "Technology", ["小米", "Xiaomi Corp", "1810"]),
    StockConfig("1299.HK", "AIA Group", "HKEX", "Financial Services", ["友邦保險", "AIA", "1299"]),
    StockConfig("005930.KS", "Samsung Electronics", "KRX", "Technology", ["삼성전자", "Samsung", "005930"]),
    StockConfig("000660.KS", "SK Hynix", "KRX", "Technology", ["SK하이닉스", "하이닉스", "000660"]),
    StockConfig("035420.KS", "NAVER", "KRX", "Communication Services", ["네이버", "Naver Corp", "035420"]),
    StockConfig("005380.KS", "Hyundai Motor", "KRX", "Consumer Cyclical", ["현대차", "현대자동차", "005380"]),
    StockConfig("035720.KS", "Kakao", "KRX", "Communication Services", ["카카오", "Kakao Corp", "035720"]),
]


class Settings:
    """Application settings loaded securely from environment variables with safe SQLite fallback."""

    def __init__(self) -> None:
        _load_env_files()
        self.apify_api_token: str = os.getenv("APIFY_API_TOKEN", "").strip()
        self.exa_api_key: str = os.getenv("EXA_API_KEY", "").strip()

        raw_db = os.getenv("DATABASE_URL", "").strip()
        default_sqlite = f"sqlite:///{BASE_DIR / 'data' / 'hengseng.db'}"
        self.database_url: str = raw_db if raw_db else default_sqlite
        self.using_sqlite_fallback: bool = not bool(raw_db) or self.database_url.startswith("sqlite")

        raw_model_path = os.getenv("MODEL_PATH", "").strip()
        self.model_path: Path = Path(raw_model_path) if raw_model_path else (BASE_DIR / "models" / "artifacts")
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.data_dir: Path = BASE_DIR / "data"
        self.reports_dir: Path = BASE_DIR / "reports"
        self.lookback_days_social: int = int(os.getenv("LOOKBACK_DAYS_SOCIAL", "10"))
        self.lookback_days_price: int = int(os.getenv("LOOKBACK_DAYS_PRICE", "90"))
        self.sentiment_model_name: str = os.getenv(
            "SENTIMENT_MODEL_NAME",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        )
        self.tickers_config_path: Path = BASE_DIR / "config" / "tickers.json"
        self.stocks: List[StockConfig] = self._load_stocks()

    @property
    def has_apify(self) -> bool:
        return bool(self.apify_api_token)

    @property
    def has_exa(self) -> bool:
        return bool(self.exa_api_key)

    def validate_configuration(self) -> Dict[str, Any]:
        """Return safe configuration diagnostics without ever exposing credential values."""
        checks = {
            "APIFY_API_TOKEN": {
                "configured": self.has_apify,
                "required": False,
                "status": "configured" if self.has_apify else "missing",
                "message": (
                    "APIFY_API_TOKEN is loaded for live Apify actor runs."
                    if self.has_apify
                    else "APIFY_API_TOKEN is not set. Social collectors cannot access live Apify actors and will fall back to labeled SIMULATED DEMO DATA."
                ),
            },
            "EXA_API_KEY": {
                "configured": self.has_exa,
                "required": False,
                "status": "configured" if self.has_exa else "missing",
                "message": (
                    "EXA_API_KEY is loaded for live Exa financial news search."
                    if self.has_exa
                    else "EXA_API_KEY is not set. Financial news agent cannot query Exa Search API and will fall back to labeled SIMULATED DEMO DATA."
                ),
            },
            "DATABASE_URL": {
                "configured": bool(os.getenv("DATABASE_URL", "").strip()),
                "required": False,
                "status": "sqlite_fallback" if self.using_sqlite_fallback else "configured",
                "message": (
                    f"Using local SQLite database fallback ({self.data_dir.name}/hengseng.db)."
                    if self.using_sqlite_fallback
                    else "External SQL database URL configured."
                ),
            },
            "MODEL_PATH": {
                "configured": bool(os.getenv("MODEL_PATH", "").strip()),
                "required": False,
                "status": "ready",
                "message": f"Model artifact directory ready ({self.model_path.name}).",
            },
        }
        return {
            "valid": True,
            "live_social_ready": self.has_apify,
            "live_news_ready": self.has_exa,
            "checks": checks,
        }

    def _load_stocks(self) -> List[StockConfig]:
        if self.tickers_config_path.exists():
            try:
                raw = json.loads(self.tickers_config_path.read_text(encoding="utf-8"))
                return [StockConfig(**item) for item in raw if item.get("enabled", True)]
            except Exception:
                return list(DEFAULT_STOCKS)
        return list(DEFAULT_STOCKS)

    def save_stocks(self, stocks: List[StockConfig]) -> None:
        self.tickers_config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "ticker": s.ticker,
                "company_name": s.company_name,
                "exchange": s.exchange,
                "sector": s.sector,
                "aliases": s.aliases,
                "enabled": s.enabled,
            }
            for s in stocks
        ]
        self.tickers_config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.stocks = stocks


_CACHED_SETTINGS: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is None or reload:
        _CACHED_SETTINGS = Settings()
    return _CACHED_SETTINGS
