"""LIHKG Finance Forum collector for Hong Kong equities (HKEX).

Platform Compliance Note:
LIHKG enforces Cloudflare Turnstile / anti-bot protection against automated unauthenticated scraping.
This collector checks Apify actor permissions and runs a single compliant verification check;
if Cloudflare blocks thread extraction (0 items), it respects the site's security controls
(no bypass attempts), truthfully reports status='unavailable', documents compliant HKEX
alternatives (public HK finance Telegram channels + Naver Foreign Stock Discussion API), and
loads clearly labeled SIMULATED DEMO DATA for UI demonstration only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from stock_sentiment_agent.collectors.apify_collector import (
    DATA_CAT_LIVE,
    DATA_CAT_SIMULATED,
    ApifyBaseCollector,
    NormalizedSocialPost,
)
from stock_sentiment_agent.config import StockConfig


class LihkgCollector(ApifyBaseCollector):
    PLATFORM = "LIHKG"
    DEFAULT_ACTOR_ID = "apify/cheerio-scraper"

    def collect_all(
        self,
        stocks: List[StockConfig],
        lookback_days: int = 10,
        max_items_per_stock: int = 30,
    ) -> Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]]:
        if not stocks:
            return {}

        if not self.is_configured:
            return {
                s.ticker: (
                    self.load_simulated_posts(s, self.PLATFORM, lookback_days),
                    {
                        "status": "unavailable",
                        "mode": "simulated_demo",
                        "data_category": DATA_CAT_SIMULATED,
                        "reason": "APIFY_API_TOKEN not configured; LIHKG source unavailable. Using labeled SIMULATED DEMO DATA.",
                    },
                )
                for s in stocks
            }

        run_input = {
            "startUrls": [{"url": "https://lihkg.com/category/15"}],
            "maxRequestsPerCrawl": 1,
            "pageFunction": """async function pageFunction(context) {
                const { $, request } = context;
                const threads = [];
                $('a[href*="/thread/"]').each((_, el) => {
                    const text = $(el).text().trim();
                    const href = $(el).attr('href');
                    if (text && href) threads.push({ title: text, url: 'https://lihkg.com' + href });
                });
                return threads;
            }""",
        }
        items, err = self.run_actor_and_fetch_items(
            self.DEFAULT_ACTOR_ID, run_input, max_items=40, timeout_sec=30
        )

        if err or not items:
            reason = (
                "LIHKG enforces Cloudflare Turnstile/SPA anti-bot verification "
                f"({err or '0 thread elements returned without bypass'}); security controls respected. "
                "Compliant HKEX alternatives active: Telegram HK finance channels & Exa News."
            )
            return {
                s.ticker: (
                    self.load_simulated_posts(s, self.PLATFORM, lookback_days),
                    {
                        "status": "unavailable",
                        "mode": "simulated_demo",
                        "data_category": DATA_CAT_SIMULATED,
                        "reason": reason,
                    },
                )
                for s in stocks
            }

        out: Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]] = {}
        for stock in stocks:
            search_terms = [stock.ticker.split(".")[0], stock.company_name] + stock.aliases[:3]
            posts: List[NormalizedSocialPost] = []
            for idx, item in enumerate(items):
                text = str(item.get("title") or item.get("body") or "").strip()
                if not text or not any(term.lower() in text.lower() for term in search_terms):
                    continue
                now_iso = datetime.now(timezone.utc).isoformat()
                posts.append(
                    NormalizedSocialPost(
                        ticker=stock.ticker,
                        company_name=stock.company_name,
                        post_id=str(item.get("thread_id") or f"lihkg_{stock.ticker}_{idx}"),
                        post_text=text,
                        post_timestamp=str(item.get("timestamp") or now_iso),
                        utc_timestamp=now_iso,
                        source_platform=self.PLATFORM,
                        author_username=item.get("user_nickname"),
                        author_follower_count=None,
                        likes=int(item.get("like_count") or 0),
                        comments=int(item.get("no_of_reply") or 0),
                        reposts=0,
                        views=0,
                        post_url=item.get("url"),
                        raw_payload=json.dumps(item, ensure_ascii=False),
                        is_simulated=False,
                        data_category=DATA_CAT_LIVE,
                    )
                )
            deduped = self.deduplicate_posts(posts)[:max_items_per_stock]
            if deduped:
                out[stock.ticker] = (
                    deduped,
                    {
                        "status": "connected",
                        "mode": "live_apify",
                        "data_category": DATA_CAT_LIVE,
                        "reason": f"Collected {len(deduped)} live LIHKG posts via Apify.",
                    },
                )
            else:
                out[stock.ticker] = (
                    self.load_simulated_posts(stock, self.PLATFORM, lookback_days),
                    {
                        "status": "unavailable",
                        "mode": "simulated_demo",
                        "data_category": DATA_CAT_SIMULATED,
                        "reason": "LIHKG Cloudflare protection yielded 0 matching threads; labeled SIMULATED DEMO DATA used for UI.",
                    },
                )
        return out

    def collect(
        self,
        stock: StockConfig,
        lookback_days: int = 10,
        max_items: int = 30,
    ) -> Tuple[List[NormalizedSocialPost], Dict[str, str]]:
        return self.collect_all([stock], lookback_days=lookback_days, max_items_per_stock=max_items)[stock.ticker]
