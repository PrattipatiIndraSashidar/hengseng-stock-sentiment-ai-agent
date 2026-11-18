"""Database connection manager supporting SQLAlchemy (PostgreSQL/SQLite) and stdlib sqlite3 fallback."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from stock_sentiment_agent.config import get_settings

SQLITE_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    company_name VARCHAR(120) NOT NULL,
    exchange VARCHAR(16) NOT NULL,
    sector VARCHAR(80),
    currency VARCHAR(8) NOT NULL DEFAULT 'HKD',
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_stocks_ticker ON stocks(ticker);
CREATE INDEX IF NOT EXISTS ix_stocks_exchange ON stocks(exchange);

CREATE TABLE IF NOT EXISTS social_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    company_name VARCHAR(120) NOT NULL,
    post_id VARCHAR(128) NOT NULL,
    post_text TEXT NOT NULL,
    cleaned_text TEXT,
    post_timestamp VARCHAR(64) NOT NULL,
    utc_timestamp TEXT NOT NULL,
    aligned_trading_date VARCHAR(16),
    source_platform VARCHAR(40) NOT NULL,
    author_username VARCHAR(120),
    author_follower_count INTEGER,
    likes INTEGER NOT NULL DEFAULT 0,
    comments INTEGER NOT NULL DEFAULT 0,
    reposts INTEGER NOT NULL DEFAULT 0,
    views INTEGER NOT NULL DEFAULT 0,
    post_url VARCHAR(512),
    raw_payload TEXT,
    is_simulated BOOLEAN NOT NULL DEFAULT 0,
    collected_at TEXT NOT NULL,
    CONSTRAINT uq_social_platform_post UNIQUE (source_platform, post_id)
);
CREATE INDEX IF NOT EXISTS ix_social_ticker_utc ON social_posts(ticker, utc_timestamp);
CREATE INDEX IF NOT EXISTS ix_social_aligned_date ON social_posts(aligned_trading_date);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    headline VARCHAR(400) NOT NULL,
    article_url VARCHAR(600) NOT NULL UNIQUE,
    published_at VARCHAR(64),
    published_utc TEXT NOT NULL,
    aligned_trading_date VARCHAR(16),
    summary TEXT,
    source VARCHAR(120),
    relevance_score REAL NOT NULL DEFAULT 0.0,
    sentiment_label VARCHAR(16),
    sentiment_score REAL,
    is_simulated BOOLEAN NOT NULL DEFAULT 0,
    collected_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_news_ticker_pub ON news_articles(ticker, published_utc);

CREATE TABLE IF NOT EXISTS stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    trading_date VARCHAR(16) NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    daily_return REAL,
    is_simulated BOOLEAN NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    CONSTRAINT uq_price_ticker_date UNIQUE (ticker, trading_date)
);
CREATE INDEX IF NOT EXISTS ix_price_ticker_date ON stock_prices(ticker, trading_date);

CREATE TABLE IF NOT EXISTS sentiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    trading_date VARCHAR(16) NOT NULL,
    source_scope VARCHAR(40) NOT NULL,
    total_items INTEGER NOT NULL DEFAULT 0,
    positive_count INTEGER NOT NULL DEFAULT 0,
    negative_count INTEGER NOT NULL DEFAULT 0,
    neutral_count INTEGER NOT NULL DEFAULT 0,
    positive_pct REAL NOT NULL DEFAULT 0.0,
    negative_pct REAL NOT NULL DEFAULT 0.0,
    neutral_pct REAL NOT NULL DEFAULT 0.0,
    sentiment_score REAL NOT NULL DEFAULT 0.0,
    engagement_weighted_score REAL NOT NULL DEFAULT 0.0,
    bullish_probability REAL NOT NULL DEFAULT 0.5,
    bearish_probability REAL NOT NULL DEFAULT 0.5,
    confidence_score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    CONSTRAINT uq_sentiment_scope UNIQUE (ticker, trading_date, source_scope)
);
CREATE INDEX IF NOT EXISTS ix_sentiment_ticker_date ON sentiment_results(ticker, trading_date);

CREATE TABLE IF NOT EXISTS financial_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    trading_date VARCHAR(16) NOT NULL,
    social_mentions INTEGER NOT NULL DEFAULT 0,
    social_sentiment REAL NOT NULL DEFAULT 0.0,
    pos_neg_ratio REAL NOT NULL DEFAULT 1.0,
    engagement_score REAL NOT NULL DEFAULT 0.0,
    avg_author_followers REAL,
    mention_growth REAL NOT NULL DEFAULT 0.0,
    sentiment_change REAL NOT NULL DEFAULT 0.0,
    news_sentiment REAL NOT NULL DEFAULT 0.0,
    news_count INTEGER NOT NULL DEFAULT 0,
    news_relevance REAL NOT NULL DEFAULT 0.0,
    news_recency REAL NOT NULL DEFAULT 0.0,
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    sma_5 REAL,
    sma_10 REAL,
    sma_20 REAL,
    volatility_10 REAL,
    support_level REAL,
    resistance_level REAL,
    target_next_close REAL,
    created_at TEXT NOT NULL,
    CONSTRAINT uq_feature_ticker_date UNIQUE (ticker, trading_date)
);
CREATE INDEX IF NOT EXISTS ix_feature_ticker_date ON financial_features(ticker, trading_date);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    prediction_date VARCHAR(16) NOT NULL,
    forecast_target_date VARCHAR(16) NOT NULL,
    last_close REAL NOT NULL,
    predicted_close REAL NOT NULL,
    actual_close REAL,
    predicted_pct_change REAL NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    mae REAL,
    rmse REAL,
    mape REAL,
    directional_accuracy REAL,
    is_validated BOOLEAN NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pred_ticker_target ON predictions(ticker, forecast_target_date);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    model_name VARCHAR(64) NOT NULL,
    split_window VARCHAR(64) NOT NULL,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    mape REAL NOT NULL,
    directional_accuracy REAL NOT NULL,
    is_best_model BOOLEAN NOT NULL DEFAULT 0,
    feature_importances_json TEXT,
    evaluated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_metrics_ticker_model ON model_metrics(ticker, model_name);

CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name VARCHAR(64) NOT NULL,
    status VARCHAR(24) NOT NULL,
    ticker VARCHAR(20),
    message TEXT NOT NULL,
    records_processed INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_agent_logs_name ON agent_logs(agent_name);
"""


class DatabaseManager:
    """Unified DB manager using SQLAlchemy when installed or sqlite3 stdlib directly."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        self.sqlite_path = self._resolve_sqlite_path(self.database_url)
        if self.sqlite_path:
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._engine = None
        self._session_factory = None
        self._init_sqlalchemy()

    @staticmethod
    def _resolve_sqlite_path(url: str) -> Optional[str]:
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "", 1)
        if url == ":memory:":
            return ":memory:"
        return None

    def _init_sqlalchemy(self) -> None:
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from stock_sentiment_agent.database.schema import Base

            if Base is not None:
                connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
                self._engine = create_engine(self.database_url, connect_args=connect_args, future=True)
                self._session_factory = sessionmaker(bind=self._engine, autoflush=False, autocommit=False)
        except ImportError:
            self._engine = None
            self._session_factory = None

    def init_tables(self) -> None:
        if self._engine is not None:
            from stock_sentiment_agent.database.schema import Base

            Base.metadata.create_all(bind=self._engine)
        elif self.sqlite_path is not None:
            with self.sqlite_connection() as conn:
                conn.executescript(SQLITE_DDL)
        else:
            raise RuntimeError("SQLAlchemy is required for non-SQLite database URLs.")

    @contextmanager
    def sqlite_connection(self):
        if self.sqlite_path is None:
            raise RuntimeError("Direct sqlite_connection requires a SQLite DATABASE_URL.")
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def session(self):
        if self._session_factory is None:
            raise RuntimeError("SQLAlchemy is not installed; use sqlite_connection().")
        sess = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def seed_stocks(self, stocks: Iterable[Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        if self.sqlite_path is not None:
            with self.sqlite_connection() as conn:
                for s in stocks:
                    currency = "HKD" if s.exchange == "HKEX" else "KRW"
                    conn.execute(
                        """
                        INSERT INTO stocks (ticker, company_name, exchange, sector, currency, is_active, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ticker) DO UPDATE SET
                            company_name=excluded.company_name,
                            exchange=excluded.exchange,
                            sector=excluded.sector,
                            is_active=excluded.is_active
                        """,
                        (s.ticker, s.company_name, s.exchange, s.sector, currency, int(s.enabled), now),
                    )
                    count += 1
        return count

    def log_agent(
        self,
        agent_name: str,
        status: str,
        message: str,
        ticker: Optional[str] = None,
        records_processed: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if self.sqlite_path is not None:
            with self.sqlite_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_logs (agent_name, status, ticker, message, records_processed, duration_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (agent_name, status, ticker, message, records_processed, duration_ms, now),
                )

    def query_rows(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        if self.sqlite_path is not None:
            with self.sqlite_connection() as conn:
                cur = conn.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
        return []


_DEFAULT_DB: Optional[DatabaseManager] = None


def get_db_manager(database_url: Optional[str] = None) -> DatabaseManager:
    global _DEFAULT_DB
    if database_url is not None:
        return DatabaseManager(database_url)
    if _DEFAULT_DB is None:
        _DEFAULT_DB = DatabaseManager()
    return _DEFAULT_DB
