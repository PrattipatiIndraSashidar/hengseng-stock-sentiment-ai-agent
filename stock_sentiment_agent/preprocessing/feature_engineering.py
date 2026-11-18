"""Financial technical indicators, support/resistance extraction, and aligned ML feature pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from stock_sentiment_agent.preprocessing.timestamp_alignment import (
    align_timestamp_to_trading_session,
    parse_utc_timestamp,
)


@dataclass
class AlignedFeatureRow:
    ticker: str
    trading_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: float
    rsi_14: float
    macd: float
    macd_signal: float
    sma_5: float
    sma_10: float
    sma_20: float
    volatility_10: float
    support_level: float
    resistance_level: float
    close_to_sma5: float
    close_to_sma20: float
    intraday_range: float
    macd_diff_norm: float
    social_mentions: int
    social_sentiment: float
    pos_neg_ratio: float
    engagement_score: float
    avg_author_followers: float
    mention_growth: float
    sentiment_change: float
    news_sentiment: float
    news_count: int
    news_relevance: float
    news_recency: float
    target_next_close: Optional[float]


def _ema(values: Sequence[float], span: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def compute_technical_indicators(price_bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute RSI-14, MACD(12,26,9), SMA-5/10/20, 10d volatility, and Support/Resistance.

    Support/Resistance method:
    Combines classic floor-trader Pivot Point bounds (2*P - H, 2*P - L) with 20-session
    rolling swing low/high percentiles using only observations <= t (strictly causal).
    """
    if not price_bars:
        return []

    bars = sorted(price_bars, key=lambda b: b["trading_date"])
    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = _ema(macd_line, 9)

    enriched: List[Dict[str, Any]] = []
    returns: List[float] = []

    for i, bar in enumerate(bars):
        c = closes[i]
        prev_c = closes[i - 1] if i > 0 else c
        ret = (c - prev_c) / prev_c if i > 0 and prev_c > 0 else 0.0
        returns.append(ret)

        win_start = max(1, i - 13)
        gains, losses = [], []
        for j in range(win_start, i + 1):
            diff = closes[j] - closes[j - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss == 0:
            rsi = 65.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        sma5 = sum(closes[max(0, i - 4) : i + 1]) / min(i + 1, 5)
        sma10 = sum(closes[max(0, i - 9) : i + 1]) / min(i + 1, 10)
        sma20 = sum(closes[max(0, i - 19) : i + 1]) / min(i + 1, 20)

        ret_win = returns[max(0, i - 9) : i + 1]
        mean_r = sum(ret_win) / len(ret_win)
        var_r = sum((r - mean_r) ** 2 for r in ret_win) / max(1, len(ret_win) - 1)
        vol10 = math.sqrt(var_r)

        swing_low = min(lows[max(0, i - 19) : i + 1])
        swing_high = max(highs[max(0, i - 19) : i + 1])
        pivot = (highs[i] + lows[i] + c) / 3.0
        pivot_s1 = 2.0 * pivot - highs[i]
        pivot_r1 = 2.0 * pivot - lows[i]
        support = round(0.5 * swing_low + 0.5 * min(pivot_s1, c * 0.99), 4)
        resistance = round(0.5 * swing_high + 0.5 * max(pivot_r1, c * 1.01), 4)

        enriched.append(
            {
                **bar,
                "daily_return": round(ret, 6),
                "rsi_14": round(rsi, 2),
                "macd": round(macd_line[i], 4),
                "macd_signal": round(signal_line[i], 4),
                "sma_5": round(sma5, 4),
                "sma_10": round(sma10, 4),
                "sma_20": round(sma20, 4),
                "volatility_10": round(vol10, 6),
                "support_level": support,
                "resistance_level": resistance,
                "close_to_sma5": round((c - sma5) / max(sma5, 1e-8), 6),
                "close_to_sma20": round((c - sma20) / max(sma20, 1e-8), 6),
                "intraday_range": round((highs[i] - lows[i]) / max(c, 1e-8), 6),
                "macd_diff_norm": round((macd_line[i] - signal_line[i]) / max(c, 1e-8), 6),
            }
        )
    return enriched


class FeatureEngineer:
    """Aligns social, news, and OHLCV data to build leak-free n+1 feature matrices."""

    # Stationary technical features used when live social/news feeds are unavailable
    PRICE_FEATURE_COLUMNS = [
        "daily_return",
        "rsi_14",
        "close_to_sma5",
        "close_to_sma20",
        "intraday_range",
        "macd_diff_norm",
        "volatility_10",
    ]

    # Full multimodal features used ONLY when genuine (non-simulated) social or news signals are available
    FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS + [
        "social_sentiment",
        "pos_neg_ratio",
        "engagement_score",
        "sentiment_change",
        "news_sentiment",
        "news_relevance",
        "news_recency",
    ]

    @classmethod
    def select_feature_columns(cls, has_genuine_social: bool, has_genuine_news: bool) -> List[str]:
        cols = list(cls.PRICE_FEATURE_COLUMNS)
        if has_genuine_social:
            cols.extend(["social_sentiment", "pos_neg_ratio", "engagement_score", "sentiment_change"])
        if has_genuine_news:
            cols.extend(["news_sentiment", "news_relevance", "news_recency"])
        return cols

    def build_features(
        self,
        ticker: str,
        exchange: str,
        price_bars: List[Dict[str, Any]],
        scored_social_posts: List[Dict[str, Any]],
        scored_news: List[Dict[str, Any]],
    ) -> List[AlignedFeatureRow]:
        tech_bars = compute_technical_indicators(price_bars)
        if not tech_bars:
            return []

        # Group social posts & news by aligned trading session
        social_by_day: Dict[str, List[Dict[str, Any]]] = {}
        for post in scored_social_posts:
            day = post.get("aligned_trading_date") or align_timestamp_to_trading_session(
                post.get("utc_timestamp", ""), exchange
            )
            social_by_day.setdefault(day, []).append(post)

        news_by_day: Dict[str, List[Dict[str, Any]]] = {}
        for art in scored_news:
            day = art.get("aligned_trading_date") or align_timestamp_to_trading_session(
                art.get("published_utc", ""), exchange
            )
            news_by_day.setdefault(day, []).append(art)

        rows: List[AlignedFeatureRow] = []
        prev_mentions = 0
        prev_sentiment = 0.0

        for i, bar in enumerate(tech_bars):
            t_date = bar["trading_date"]
            day_posts = social_by_day.get(t_date, [])
            day_news = news_by_day.get(t_date, [])

            mentions = len(day_posts)
            if mentions > 0:
                total_w = sum(float(p.get("engagement_weight", 1.0)) for p in day_posts)
                social_sent = (
                    sum(float(p.get("sentiment_score", 0.0)) * float(p.get("engagement_weight", 1.0)) for p in day_posts)
                    / max(total_w, 1e-6)
                )
                pos_cnt = sum(1 for p in day_posts if p.get("sentiment_label") == "Positive")
                neg_cnt = sum(1 for p in day_posts if p.get("sentiment_label") == "Negative")
                pos_neg_ratio = (pos_cnt + 1.0) / (neg_cnt + 1.0)
                eng_score = sum(
                    math.log1p(int(p.get("likes", 0)) + 2 * int(p.get("comments", 0)) + 3 * int(p.get("reposts", 0)))
                    for p in day_posts
                ) / mentions
                followers = [int(p["author_follower_count"]) for p in day_posts if p.get("author_follower_count")]
                avg_followers = sum(followers) / len(followers) if followers else 0.0
            else:
                # Carry forward decayed prior sentiment when no new social items occurred on that session
                social_sent = prev_sentiment * 0.85
                pos_neg_ratio = 1.0
                eng_score = 0.0
                avg_followers = 0.0

            mention_growth = (mentions - prev_mentions) / max(prev_mentions, 1)
            sentiment_change = social_sent - prev_sentiment
            prev_mentions = mentions if mentions > 0 else prev_mentions
            prev_sentiment = social_sent

            news_count = len(day_news)
            if news_count > 0:
                news_sent = sum(float(n.get("sentiment_score", 0.0)) for n in day_news) / news_count
                news_rel = sum(float(n.get("relevance_score", 0.7)) for n in day_news) / news_count
                session_end = parse_utc_timestamp(f"{t_date}T16:00:00+00:00")
                recencies = []
                for n in day_news:
                    pub_dt = parse_utc_timestamp(str(n.get("published_utc", t_date)))
                    age_hrs = max(0.0, (session_end - pub_dt).total_seconds() / 3600.0)
                    recencies.append(math.exp(-age_hrs / 48.0))
                news_recency = sum(recencies) / len(recencies)
            else:
                news_sent = 0.0
                news_rel = 0.0
                news_recency = 0.0

            next_close = float(tech_bars[i + 1]["close"]) if i + 1 < len(tech_bars) else None

            rows.append(
                AlignedFeatureRow(
                    ticker=ticker,
                    trading_date=t_date,
                    open=float(bar["open"]),
                    high=float(bar["high"]),
                    low=float(bar["low"]),
                    close=float(bar["close"]),
                    volume=float(bar["volume"]),
                    daily_return=float(bar["daily_return"]),
                    rsi_14=float(bar["rsi_14"]),
                    macd=float(bar["macd"]),
                    macd_signal=float(bar["macd_signal"]),
                    sma_5=float(bar["sma_5"]),
                    sma_10=float(bar["sma_10"]),
                    sma_20=float(bar["sma_20"]),
                    volatility_10=float(bar["volatility_10"]),
                    support_level=float(bar["support_level"]),
                    resistance_level=float(bar["resistance_level"]),
                    close_to_sma5=float(bar["close_to_sma5"]),
                    close_to_sma20=float(bar["close_to_sma20"]),
                    intraday_range=float(bar["intraday_range"]),
                    macd_diff_norm=float(bar["macd_diff_norm"]),
                    social_mentions=mentions,
                    social_sentiment=round(social_sent, 4),
                    pos_neg_ratio=round(pos_neg_ratio, 4),
                    engagement_score=round(eng_score, 4),
                    avg_author_followers=round(avg_followers, 2),
                    mention_growth=round(mention_growth, 4),
                    sentiment_change=round(sentiment_change, 4),
                    news_sentiment=round(news_sent, 4),
                    news_count=news_count,
                    news_relevance=round(news_rel, 4),
                    news_recency=round(news_recency, 4),
                    target_next_close=next_close,
                )
            )
        return rows

    @staticmethod
    def to_dicts(rows: List[AlignedFeatureRow]) -> List[Dict[str, Any]]:
        return [asdict(r) for r in rows]
