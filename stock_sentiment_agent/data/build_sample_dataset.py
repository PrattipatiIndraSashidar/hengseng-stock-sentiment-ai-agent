"""Generates realistic, clearly labeled demonstration dataset for HKEX and KRX equities."""

from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE_SPECS = [
    ("0700.HK", "Tencent", "HKEX", 412.0, 22_000_000, 0.0018, "bull"),
    ("9988.HK", "Alibaba", "HKEX", 88.5, 38_000_000, 0.0008, "bull"),
    ("3690.HK", "Meituan", "HKEX", 174.0, 29_000_000, -0.0006, "bear"),
    ("1810.HK", "Xiaomi", "HKEX", 28.4, 95_000_000, 0.0024, "bull"),
    ("1299.HK", "AIA Group", "HKEX", 59.2, 19_000_000, -0.0002, "neutral"),
    ("005930.KS", "Samsung Electronics", "KRX", 57800.0, 16_500_000, -0.0010, "bear"),
    ("000660.KS", "SK Hynix", "KRX", 186500.0, 4_200_000, 0.0026, "bull"),
    ("035420.KS", "NAVER", "KRX", 192400.0, 980_000, 0.0011, "bull"),
    ("005380.KS", "Hyundai Motor", "KRX", 218000.0, 740_000, 0.0005, "neutral"),
    ("035720.KS", "Kakao", "KRX", 38900.0, 1_900_000, -0.0014, "bear"),
]

HK_POSTS = {
    "bull": [
        ("{cn} ({tk}) 業績亮眼，雲業務同AI廣告增長超預期，已經加倉做多。", "LIHKG", 84, 21, 0, 1420),
        ("{cn} strong buyback momentum and gaming revenue beat estimates. Going long above support.", "Telegram", 125, 19, 34, 4800),
        ("資金流入{cn}，技術面突破20天線，短線看漲買進。", "LIHKG", 52, 14, 0, 980),
        ("{cn} Q3 operating margin expansion looks solid; accumulating on dips.", "Telegram", 67, 11, 18, 2900),
    ],
    "bear": [
        ("{cn} ({tk}) 面對激烈競爭同監管風險，業績不及預期，暫時減倉做空。", "LIHKG", 71, 29, 0, 1650),
        ("{cn} margin pressure continues; trimming exposure ahead of guidance revision.", "Telegram", 48, 16, 12, 2300),
        ("成交縮量，{cn}走勢疲弱，睇淡後市表現。", "LIHKG", 39, 12, 0, 860),
        ("{cn} regulatory headwind weighing on multiples; short position into resistance.", "Telegram", 41, 9, 11, 2100),
    ],
    "neutral": [
        ("{cn} ({tk}) 走勢橫行整固，暫時觀望等待業績公布。", "LIHKG", 28, 8, 0, 620),
        ("{cn} holding position inside range; waiting for clearer macro catalyst.", "Telegram", 33, 7, 5, 1700),
        ("{cn}估值合理但缺乏短期催化劑，繼續持有觀望。", "LIHKG", 24, 6, 0, 510),
        ("{cn} steady dividend profile, wait and see on capital allocation update.", "Telegram", 29, 5, 4, 1400),
    ],
}

KR_POSTS = {
    "bull": [
        ("{cn} ({tk}) HBM 수주 확대 및 실적 개선 기대감으로 외국인 매수세 유입! 추가 매수.", "Naver Finance", 96, 24, 0, 2150),
        ("{cn} 어닝 서프라이즈 전망, 목표가 상향 리포트 연달아 나옴. 강세 돌파 기대.", "Daum Finance", 64, 15, 7, 1490),
        ("{cn} AI 반도체 수요 급등 수혜주, 저평가 구간이라 분할 매수 유지.", "Naver Finance", 78, 18, 0, 1880),
        ("{cn} strong institutional accumulation ahead of earnings release; bullish breakout.", "Daum Finance", 49, 10, 6, 1120),
    ],
    "bear": [
        ("{cn} ({tk}) 실적 부진 우려와 공매도 증가로 단기 약세 이탈 주의, 비중 축소.", "Naver Finance", 82, 31, 0, 2400),
        ("{cn} 수익성 둔화 및 규제 리스크 부각... 당분간 매도 관점 유지.", "Daum Finance", 57, 19, 8, 1620),
        ("{cn} 목표가 하향 조정 나오면서 투자심리 위축, 손절 고민 중.", "Naver Finance", 65, 22, 0, 1950),
        ("{cn} weak memory pricing and macro headwind; short-term downside risk.", "Daum Finance", 44, 13, 5, 1210),
    ],
    "neutral": [
        ("{cn} ({tk}) 박스권 등락 지속, 실적 발표 전까지 관망 유지.", "Naver Finance", 31, 9, 0, 910),
        ("{cn} 배당 매력은 있으나 모멘텀 부족으로 보유 유지.", "Daum Finance", 27, 7, 2, 780),
        ("{cn} 외국인 수급 혼조세, 추가 방향성 확인 후 대응 예정.", "Naver Finance", 23, 6, 0, 690),
        ("{cn} trading flat near 20-day moving average; hold rating consensus.", "Daum Finance", 21, 5, 3, 640),
    ],
}


def build_dataset() -> dict:
    rng = random.Random(202611)
    # Generate 38 weekdays ending on 2026-11-16
    end_date = date(2026, 11, 16)
    trading_days = []
    cur = end_date - timedelta(days=62)
    while cur <= end_date:
        if cur.weekday() < 5:
            trading_days.append(cur)
        cur += timedelta(days=1)
    trading_days = trading_days[-38:]

    stock_prices = []
    social_posts = []
    news_articles = []

    for ticker, name, exchange, start_price, avg_vol, drift, bias in BASE_SPECS:
        price = start_price
        for idx, d in enumerate(trading_days):
            wave = math.sin(idx * 0.35 + len(ticker)) * 0.008
            ret = drift + wave + rng.gauss(0.0, 0.011)
            open_p = price * (1.0 + rng.gauss(0.0, 0.004))
            close_p = max(1.0, price * (1.0 + ret))
            high_p = max(open_p, close_p) * (1.0 + abs(rng.gauss(0.006, 0.004)))
            low_p = min(open_p, close_p) * (1.0 - abs(rng.gauss(0.006, 0.004)))
            vol = avg_vol * (0.75 + rng.random() * 0.65)
            decimals = 2 if exchange == "HKEX" else 0
            stock_prices.append(
                {
                    "ticker": ticker,
                    "trading_date": d.isoformat(),
                    "open": round(open_p, decimals),
                    "high": round(high_p, decimals),
                    "low": round(low_p, decimals),
                    "close": round(close_p, decimals),
                    "volume": int(vol),
                }
            )
            price = close_p

        # Generate 10 days of social posts across the last 10 trading days
        recent_days = trading_days[-10:]
        templates = HK_POSTS if exchange == "HKEX" else KR_POSTS
        post_pool = templates[bias] + templates["neutral"][:2]
        for day_idx, d in enumerate(recent_days):
            for p_idx in range(2):
                tpl, platform, base_likes, base_comm, base_rep, base_views = post_pool[(day_idx + p_idx) % len(post_pool)]
                hour_utc = 3 + ((day_idx * 3 + p_idx * 5) % 10)
                ts_utc = datetime(d.year, d.month, d.day, hour_utc, 15, tzinfo=timezone.utc)
                text = tpl.format(cn=name, tk=ticker)
                followers = (1200 + rng.randint(100, 18000)) if platform == "Telegram" else None
                social_posts.append(
                    {
                        "ticker": ticker,
                        "company_name": name,
                        "post_id": f"sim_{ticker.replace('.', '_')}_{d.isoformat()}_{p_idx}",
                        "post_text": text,
                        "post_timestamp": ts_utc.isoformat(),
                        "utc_timestamp": ts_utc.isoformat(),
                        "source_platform": platform,
                        "author_username": f"analyst_{platform.lower().replace(' ', '')}_{day_idx}{p_idx}" if platform == "Telegram" else None,
                        "author_follower_count": followers,
                        "likes": max(0, base_likes + rng.randint(-12, 25)),
                        "comments": max(0, base_comm + rng.randint(-4, 10)),
                        "reposts": max(0, base_rep + rng.randint(-2, 8)) if platform in ("Telegram", "Daum Finance") else 0,
                        "views": max(100, base_views + rng.randint(-200, 500)),
                        "post_url": f"https://example.invalid/{platform.lower().replace(' ', '-')}/{ticker}/{d.isoformat()}/{p_idx}",
                    }
                )

        # Generate 3 financial news articles per stock
        news_templates = {
            "bull": [
                (f"{name} ({ticker}) beats quarterly revenue consensus on AI and cloud expansion",
                 f"Quarterly financial results for {name} showed accelerating operating profit, strong margin recovery, and upgraded analyst price targets.",
                 "reuters.com", 0.92),
                (f"Institutional inflows lift {name} following new product announcement and buyback plan",
                 f"Market trends point to resilient demand and bullish institutional positioning in {name} ahead of next quarter.",
                 "bloomberg.com", 0.88),
                (f"Analysts reiterate Outperform on {name} citing disciplined capital investments",
                 f"Earnings forecasts were revised higher after management detailed multi-year AI infrastructure investments.",
                 "ft.com", 0.85),
            ],
            "bear": [
                (f"{name} ({ticker}) faces margin headwinds amid intensifying sector competition",
                 f"Recent financial results reflect rising cost pressure and regulatory scrutiny, prompting cautious analyst forecasts.",
                 "reuters.com", 0.90),
                (f"Analysts trim earnings estimates for {name} on slower demand recovery",
                 f"Revenue growth lagged consensus while competitive discounting weighed on near-term profitability.",
                 "wsj.com", 0.86),
                (f"Regulatory review adds downside risk to {name} near-term valuation",
                 f"Investors remain defensive on {name} pending clarity on compliance costs and market share trends.",
                 "nikkei.com", 0.83),
            ],
            "neutral": [
                (f"{name} ({ticker}) posts steady quarterly results in line with market forecasts",
                 f"{name} delivered stable revenue and balanced capital returns while macro uncertainty kept guidance measured.",
                 "reuters.com", 0.84),
                (f"{name} maintains steady dividend outlook amid mixed regional market trends",
                 f"Analysts hold neutral consensus on {name} as valuation balances defensive cash flows with modest growth.",
                 "scmp.com", 0.80),
                (f"Investors await upcoming earnings call from {name} for 2027 guidance",
                 f"Trading volume remained steady as market participants wait for updated margin and investment targets.",
                 "bloomberg.com", 0.78),
            ],
        }
        for n_idx, (headline, summary, src, rel) in enumerate(news_templates[bias]):
            pub_day = recent_days[-1 - n_idx * 2]
            pub_dt = datetime(pub_day.year, pub_day.month, pub_day.day, 6, 30, tzinfo=timezone.utc)
            news_articles.append(
                {
                    "ticker": ticker,
                    "headline": headline,
                    "article_url": f"https://{src}/markets/{ticker.lower().replace('.', '-')}-news-{n_idx}",
                    "published_at": pub_dt.isoformat(),
                    "published_utc": pub_dt.isoformat(),
                    "summary": summary,
                    "source": src,
                    "relevance_score": rel,
                }
            )

    return {
        "metadata": {
            "is_simulated": True,
            "notice": "Clearly labeled demonstration dataset used only when live API keys or network feeds are unavailable.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "stock_prices": stock_prices,
        "social_posts": social_posts,
        "news_articles": news_articles,
    }


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "sample_dataset.json"
    out_path.write_text(json.dumps(build_dataset(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
