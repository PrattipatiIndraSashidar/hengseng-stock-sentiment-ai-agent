"""Daum Finance Forum collector for Korean equities (KRX) via Apify web-scraper."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from stock_sentiment_agent.collectors.apify_collector import (
    DATA_CAT_LIVE,
    DATA_CAT_SIMULATED,
    ApifyBaseCollector,
    NormalizedSocialPost,
)
from stock_sentiment_agent.config import StockConfig

KST = timezone(timedelta(hours=9))


def _parse_daum_kst(raw_date: str) -> datetime:
    s = (raw_date or "").strip()
    now_kst = datetime.now(KST)
    try:
        if re.match(r"^\d{1,2}:\d{2}$", s):
            hh, mm = map(int, s.split(":"))
            dt = now_kst.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if dt > now_kst + timedelta(minutes=5):
                dt -= timedelta(days=1)
            return dt.astimezone(timezone.utc)
        if re.match(r"^\d{2}\.\d{2}\.\d{2}$", s):
            yy, mm, dd = map(int, s.split("."))
            return datetime(2000 + yy, mm, dd, 15, 0, tzinfo=KST).astimezone(timezone.utc)
        if re.match(r"^\d{4}\.\d{2}\.\d{2}$", s):
            yyyy, mm, dd = map(int, s.split("."))
            return datetime(yyyy, mm, dd, 15, 0, tzinfo=KST).astimezone(timezone.utc)
        if re.match(r"^\d{2}\.\d{2}$", s):
            mm, dd = map(int, s.split("."))
            dt = datetime(now_kst.year, mm, dd, 15, 0, tzinfo=KST)
            if dt > now_kst + timedelta(days=2):
                dt = dt.replace(year=now_kst.year - 1)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _clean_daum_title(raw_title: str) -> Tuple[str, int]:
    """Strip trailing Daum UI badges ('New') and extract trailing reply count if present."""
    txt = re.sub(r"\s+New$", "", raw_title.strip()).strip()
    comments = 0
    m = re.match(r"^(.*?)\s+(\d{1,4})$", txt)
    if m and len(m.group(1).strip()) >= 3:
        txt = m.group(1).strip()
        comments = int(m.group(2))
    return txt, comments


def _parse_int(raw: Any) -> int:
    try:
        return int(re.sub(r"[^\d]", "", str(raw or "0")) or "0")
    except Exception:
        return 0


class DaumFinanceCollector(ApifyBaseCollector):
    PLATFORM = "Daum Finance"
    DEFAULT_ACTOR_ID = "apify/web-scraper"

    def collect_all(
        self,
        stocks: List[StockConfig],
        lookback_days: int = 10,
        max_items_per_stock: int = 30,
    ) -> Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]]:
        """Collect genuine Daum Finance talk-board posts across all KRX tickers in one batched Apify run."""
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
                        "reason": "APIFY_API_TOKEN not configured; Daum Finance unavailable. Using labeled SIMULATED DEMO DATA.",
                    },
                )
                for s in stocks
            }

        start_urls = []
        code_to_stock: Dict[str, StockConfig] = {}
        for s in stocks:
            code = s.ticker.split(".")[0]
            code_to_stock[code] = s
            start_urls.append(
                {
                    "url": f"https://finance.daum.net/quotes/A{code}#talks",
                    "userData": {"code": code, "ticker": s.ticker},
                }
            )

        page_fn = """async function pageFunction(context) {
            await context.waitFor(2800);
            const code = (context.request.userData && context.request.userData.code) || '';
            const out = [];
            document.querySelectorAll('table').forEach((tbl) => {
                const header = tbl.innerText || '';
                if (!header.includes('제목') && !header.includes('작성자')) return;
                tbl.querySelectorAll('tbody tr, tr').forEach((tr) => {
                    const tds = Array.from(tr.querySelectorAll('td')).map(td => td.innerText ? td.innerText.trim() : '');
                    if (tds.length < 5) return;
                    const titleRaw = tds[0] || '';
                    const author = tds[1] || '';
                    if (!titleRaw || author === '운영자' || titleRaw.startsWith('공지')) return;
                    const a = tr.querySelector('a');
                    const href = a ? (a.getAttribute('href') || '') : '';
                    out.push({
                        code,
                        titleRaw,
                        author,
                        viewsRaw: tds[2],
                        likesRaw: tds[3],
                        dateRaw: tds[4],
                        href,
                        pageUrl: context.request.url,
                    });
                });
            });
            return out;
        }"""

        run_input = {
            "startUrls": start_urls,
            "maxRequestsPerCrawl": len(stocks) + 1,
            "waitUntil": ["networkidle2"],
            "pageFunction": page_fn,
        }
        items, err = self.run_actor_and_fetch_items(
            self.DEFAULT_ACTOR_ID,
            run_input,
            max_items=max_items_per_stock * len(stocks) * 2,
            page_size=100,
            timeout_sec=90,
        )

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        grouped: Dict[str, List[NormalizedSocialPost]] = {s.ticker: [] for s in stocks}
        for idx, item in enumerate(items):
            code = str(item.get("code") or "").strip()
            stock = code_to_stock.get(code)
            if stock is None:
                continue
            title_raw = str(item.get("titleRaw") or "").strip()
            title, comments = _clean_daum_title(title_raw)
            if not title:
                continue

            date_raw = str(item.get("dateRaw") or "")
            dt_utc = _parse_daum_kst(date_raw)
            if dt_utc < cutoff:
                continue

            href = str(item.get("href") or "").strip()
            m_id = re.search(r"talks/(\d+)", href)
            talk_id = m_id.group(1) if m_id else f"{idx}_{abs(hash(title)) % 100000}"
            base_url = f"https://finance.daum.net/quotes/A{code}"
            post_url = f"{base_url}{href}" if href.startswith("#") else (href or f"{base_url}#talks")

            grouped[stock.ticker].append(
                NormalizedSocialPost(
                    ticker=stock.ticker,
                    company_name=stock.company_name,
                    post_id=f"daum_{code}_{talk_id}",
                    post_text=title,
                    post_timestamp=date_raw or dt_utc.isoformat(),
                    utc_timestamp=dt_utc.isoformat(),
                    source_platform=self.PLATFORM,
                    author_username=str(item.get("author") or "").strip() or None,
                    author_follower_count=None,
                    likes=_parse_int(item.get("likesRaw")),
                    comments=comments,
                    reposts=0,
                    views=_parse_int(item.get("viewsRaw")),
                    post_url=post_url,
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
                        "mode": "live_apify",
                        "data_category": DATA_CAT_LIVE,
                        "reason": f"Collected {len(deduped)} genuine Daum Finance posts via {self.DEFAULT_ACTOR_ID}.",
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
                        "reason": f"Daum Finance unavailable ({err or '0 rows extracted'}); loaded labeled SIMULATED DEMO DATA.",
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
