"""Naver Finance Stock Discussion Board collector for Korean equities (KRX)."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from stock_sentiment_agent.collectors.apify_collector import (
    DATA_CAT_LIVE,
    DATA_CAT_SIMULATED,
    ApifyBaseCollector,
    NormalizedSocialPost,
)
from stock_sentiment_agent.config import StockConfig

KST = timezone(timedelta(hours=9))


def _parse_naver_kst(written_at: str) -> datetime:
    raw = (written_at or "").strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


class NaverFinanceCollector(ApifyBaseCollector):
    PLATFORM = "Naver Finance"
    DEFAULT_ACTOR_ID = "apify/cheerio-scraper"

    def collect_all(
        self,
        stocks: List[StockConfig],
        lookback_days: int = 10,
        max_items_per_stock: int = 45,
    ) -> Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]]:
        """Collect genuine Naver Finance KRX posts across all KRX tickers in one batched Apify run."""
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
                        "reason": "APIFY_API_TOKEN not configured; Naver Finance unavailable. Using labeled SIMULATED DEMO DATA.",
                    },
                )
                for s in stocks
            }

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        start_urls = []
        code_to_stock: Dict[str, StockConfig] = {}
        for s in stocks:
            code = s.ticker.split(".")[0]
            code_to_stock[code] = s
            start_urls.append(
                {
                    "url": (
                        f"https://m.stock.naver.com/front-api/discussion/list"
                        f"?discussionType=domesticStock&itemCode={code}&size=40"
                    ),
                    "userData": {"code": code, "ticker": s.ticker, "depth": 0},
                }
            )

        page_fn = f"""async function pageFunction(context) {{
            const {{ body, json, request, enqueueRequest }} = context;
            let payload = json;
            if (!payload && body) {{
                try {{ payload = JSON.parse(String(body)); }} catch (e) {{ payload = null; }}
            }}
            const res = payload && payload.result ? payload.result : (payload || {{}});
            const posts = Array.isArray(res.posts) ? res.posts : [];
            const code = (request.userData && request.userData.code) || '';
            const depth = (request.userData && request.userData.depth) || 0;
            const out = [];
            let oldestMs = Date.now();
            for (const p of posts) {{
                const written = p.writtenAt || '';
                const ms = written ? Date.parse(written + '+09:00') : Date.now();
                if (!isNaN(ms) && ms < oldestMs) oldestMs = ms;
                out.push({{
                    id: String(p.id || ''),
                    code: String(p.itemCode || code),
                    title: p.title || '',
                    content: p.contentSwReplacedButImg || '',
                    writtenAt: written,
                    author: (p.writer && p.writer.nickname) || null,
                    likes: Number(p.recommendCount || 0),
                    views: Number(p.viewCount || 0),
                    notLikes: Number(p.notRecommendCount || 0),
                }});
            }}
            if (res.lastOffset && depth < 3 && oldestMs > {cutoff_ms} && posts.length >= 20 && code) {{
                await enqueueRequest({{
                    url: `https://m.stock.naver.com/front-api/discussion/list?discussionType=domesticStock&itemCode=${{code}}&size=40&offset=${{encodeURIComponent(res.lastOffset)}}`,
                    userData: {{ code, depth: depth + 1 }},
                }});
            }}
            return out;
        }}"""

        run_input = {
            "startUrls": start_urls,
            "maxRequestsPerCrawl": len(stocks) * 4 + 2,
            "additionalMimeTypes": ["application/json"],
            "pageFunction": page_fn,
        }
        items, err = self.run_actor_and_fetch_items(
            self.DEFAULT_ACTOR_ID,
            run_input,
            max_items=max_items_per_stock * len(stocks) * 2,
            page_size=100,
            timeout_sec=75,
        )

        used_mode = "live_apify"
        if (err or not items):
            # Compliant direct Npay Securities JSON endpoint fallback (no anti-bot bypass)
            direct_items = self._fetch_direct_api(stocks, lookback_days=lookback_days, max_pages=3)
            if direct_items:
                items = direct_items
                err = None
                used_mode = "live_naver_api"

        grouped: Dict[str, List[NormalizedSocialPost]] = {s.ticker: [] for s in stocks}
        for item in items:
            code = str(item.get("code") or "").strip()
            stock = code_to_stock.get(code)
            if stock is None:
                continue
            post_id_raw = str(item.get("id") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            combined_text = f"{title}. {content}".strip(" .") if content and content != title else (title or content)
            if not post_id_raw or not combined_text:
                continue

            written_at = str(item.get("writtenAt") or "")
            dt_utc = _parse_naver_kst(written_at)
            if dt_utc < cutoff:
                continue

            grouped[stock.ticker].append(
                NormalizedSocialPost(
                    ticker=stock.ticker,
                    company_name=stock.company_name,
                    post_id=f"naver_{code}_{post_id_raw}",
                    post_text=combined_text[:1200],
                    post_timestamp=written_at or dt_utc.isoformat(),
                    utc_timestamp=dt_utc.isoformat(),
                    source_platform=self.PLATFORM,
                    author_username=item.get("author"),
                    author_follower_count=None,
                    likes=int(item.get("likes") or 0),
                    comments=0,
                    reposts=0,
                    views=int(item.get("views") or 0),
                    post_url=f"https://m.stock.naver.com/domestic/stock/{code}/discussion/{post_id_raw}",
                    raw_payload=json.dumps(item, ensure_ascii=False),
                    is_simulated=False,
                    data_category=DATA_CAT_LIVE,
                )
            )

        out: Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]] = {}
        for s in stocks:
            deduped = self.deduplicate_posts(grouped.get(s.ticker, []))[:max_items_per_stock]
            if deduped:
                out[s.ticker] = (
                    deduped,
                    {
                        "status": "connected",
                        "mode": used_mode,
                        "data_category": DATA_CAT_LIVE,
                        "reason": f"Collected {len(deduped)} genuine Naver Finance posts via {self.DEFAULT_ACTOR_ID} ({lookback_days}d window).",
                    },
                )
            else:
                demo = self.load_simulated_posts(s, self.PLATFORM, lookback_days)
                out[s.ticker] = (
                    demo,
                    {
                        "status": "unavailable",
                        "mode": "simulated_demo",
                        "data_category": DATA_CAT_SIMULATED,
                        "reason": f"Naver Finance returned 0 live rows ({err or 'empty'}); loaded labeled SIMULATED DEMO DATA.",
                    },
                )
        return out

    def _fetch_direct_api(
        self, stocks: List[StockConfig], lookback_days: int = 10, max_pages: int = 3
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        collected: List[Dict[str, Any]] = []
        for s in stocks:
            code = s.ticker.split(".")[0]
            offset: Optional[str] = None
            for _ in range(max_pages):
                url = (
                    f"https://m.stock.naver.com/front-api/discussion/list"
                    f"?discussionType=domesticStock&itemCode={code}&size=40"
                )
                if offset:
                    url += f"&offset={offset}"
                try:
                    req = urllib.request.Request(
                        url,
                        headers={
                            "User-Agent": "Mozilla/5.0",
                            "Referer": "https://m.stock.naver.com/",
                            "Accept": "application/json",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    res = data.get("result") or {}
                    posts = res.get("posts") or []
                    if not posts:
                        break
                    oldest = datetime.now(timezone.utc)
                    for p in posts:
                        written = str(p.get("writtenAt") or "")
                        dt = _parse_naver_kst(written)
                        if dt < oldest:
                            oldest = dt
                        collected.append(
                            {
                                "id": str(p.get("id") or ""),
                                "code": str(p.get("itemCode") or code),
                                "title": p.get("title") or "",
                                "content": p.get("contentSwReplacedButImg") or "",
                                "writtenAt": written,
                                "author": (p.get("writer") or {}).get("nickname"),
                                "likes": int(p.get("recommendCount") or 0),
                                "views": int(p.get("viewCount") or 0),
                            }
                        )
                    offset = res.get("lastOffset")
                    if not offset or oldest < cutoff:
                        break
                except Exception:
                    break
        return collected

    def collect(
        self,
        stock: StockConfig,
        lookback_days: int = 10,
        max_items: int = 45,
    ) -> Tuple[List[NormalizedSocialPost], Dict[str, str]]:
        return self.collect_all([stock], lookback_days=lookback_days, max_items_per_stock=max_items)[stock.ticker]
