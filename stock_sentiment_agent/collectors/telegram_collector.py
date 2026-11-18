"""Telegram public channel collector for Hong Kong equities via Apify."""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any, Dict, List, Tuple

from stock_sentiment_agent.collectors.apify_collector import (
    DATA_CAT_LIVE,
    DATA_CAT_SIMULATED,
    ApifyBaseCollector,
    NormalizedSocialPost,
)
from stock_sentiment_agent.config import StockConfig

SPAM_PATTERNS = ("@bluemarketer", "airdrop", "crypto pump", "casino")


class TelegramCollector(ApifyBaseCollector):
    PLATFORM = "Telegram"
    DEFAULT_ACTOR_ID = "apify/cheerio-scraper"
    PUBLIC_HK_CHANNELS = ["hongkongstocks", "hket_news", "stheadline", "hksignal", "hkexnews"]

    def collect_all(
        self,
        stocks: List[StockConfig],
        lookback_days: int = 10,
        max_items_per_stock: int = 30,
    ) -> Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]]:
        """Collect last 10 days of HK finance Telegram messages in one paginated Apify run."""
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
                        "reason": "APIFY_API_TOKEN not configured; Telegram source unavailable. Using labeled SIMULATED DEMO DATA.",
                    },
                )
                for s in stocks
            }

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        start_urls = [
            {"url": f"https://t.me/s/{ch}", "userData": {"channel": ch, "depth": 0}}
            for ch in self.PUBLIC_HK_CHANNELS
        ]
        page_fn = f"""async function pageFunction(context) {{
            const {{ $, request, enqueueRequest }} = context;
            const ch = (request.userData && request.userData.channel) || request.url.split('/').pop().split('?')[0];
            const depth = (request.userData && request.userData.depth) || 0;
            const msgs = [];
            let oldestMs = Date.now();
            let minId = Infinity;
            $('.tgme_widget_message_wrap').each((idx, el) => {{
                const text = $(el).find('.tgme_widget_message_text').text().trim();
                const date = $(el).find('time.time').attr('datetime') || '';
                const viewsText = $(el).find('.tgme_widget_message_views').text().trim();
                const postUrl = $(el).find('a.tgme_widget_message_date').attr('href') || request.url;
                const msgId = parseInt(String(postUrl).split('/').pop() || '0', 10);
                if (!isNaN(msgId) && msgId > 0 && msgId < minId) minId = msgId;
                const ms = date ? Date.parse(date) : Date.now();
                if (!isNaN(ms) && ms < oldestMs) oldestMs = ms;
                if (text) msgs.push({{ id: String(msgId || idx), text, date, viewsText, url: postUrl, channel: ch }});
            }});
            if (depth < 4 && oldestMs > {cutoff_ms} && minId < Infinity && minId > 1) {{
                await enqueueRequest({{
                    url: `https://t.me/s/${{ch}}?before=${{minId}}`,
                    userData: {{ channel: ch, depth: depth + 1 }},
                }});
            }}
            return msgs;
        }}"""

        run_input = {
            "startUrls": start_urls,
            "maxRequestsPerCrawl": len(self.PUBLIC_HK_CHANNELS) * 5,
            "pageFunction": page_fn,
        }
        items, err = self.run_actor_and_fetch_items(
            self.DEFAULT_ACTOR_ID,
            run_input,
            max_items=600,
            page_size=100,
            timeout_sec=75,
        )

        used_mode = "live_apify"
        if (err or not items):
            direct_items = self._fetch_direct_tme(lookback_days=lookback_days, max_pages=4)
            if direct_items:
                items = direct_items
                err = None
                used_mode = "live_telegram_web"

        out: Dict[str, Tuple[List[NormalizedSocialPost], Dict[str, str]]] = {}
        for stock in stocks:
            code = stock.ticker.split(".")[0]
            short_code = code.lstrip("0")
            terms = [
                stock.ticker.lower(),
                code.lower(),
                f"({code})",
                f"({short_code})",
                stock.company_name.lower(),
            ] + [a.lower() for a in stock.aliases if len(a.strip()) >= 2]

            matched: List[NormalizedSocialPost] = []
            for idx, item in enumerate(items):
                text = str(item.get("text") or "").strip()
                lower_txt = text.lower()
                if not text or any(sp in lower_txt for sp in SPAM_PATTERNS):
                    continue
                if not any(term in lower_txt for term in terms):
                    continue

                ts = str(item.get("date") or datetime.now(timezone.utc).isoformat())
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt < cutoff:
                        continue
                except Exception:
                    pass

                ch = str(item.get("channel") or "telegram")
                post_id = f"tg_{ch}_{item.get('id') or idx}"
                matched.append(
                    NormalizedSocialPost(
                        ticker=stock.ticker,
                        company_name=stock.company_name,
                        post_id=post_id,
                        post_text=text[:1500],
                        post_timestamp=ts,
                        utc_timestamp=ts,
                        source_platform=self.PLATFORM,
                        author_username=f"@{ch}",
                        author_follower_count=None,
                        likes=0,
                        comments=0,
                        reposts=0,
                        views=_parse_views(str(item.get("viewsText") or "0")),
                        post_url=str(item.get("url") or f"https://t.me/{ch}"),
                        raw_payload=json.dumps(item, ensure_ascii=False),
                        is_simulated=False,
                        data_category=DATA_CAT_LIVE,
                    )
                )

            deduped = self.deduplicate_posts(matched)[:max_items_per_stock]
            if deduped:
                out[stock.ticker] = (
                    deduped,
                    {
                        "status": "connected",
                        "mode": used_mode,
                        "data_category": DATA_CAT_LIVE,
                        "reason": f"Collected {len(deduped)} genuine HK finance Telegram posts across {len(self.PUBLIC_HK_CHANNELS)} public channels.",
                    },
                )
            else:
                demo = self.load_simulated_posts(stock, self.PLATFORM, lookback_days)
                reason = (
                    f"Telegram channels live ({len(items)} messages scanned over {lookback_days}d), 0 direct mentions of {stock.ticker}; labeled SIMULATED DEMO DATA used for UI."
                    if items and not err
                    else f"Telegram crawl unavailable ({err or '0 messages'}); using labeled SIMULATED DEMO DATA."
                )
                out[stock.ticker] = (
                    demo,
                    {
                        "status": "connected" if (items and not err) else "unavailable",
                        "mode": "simulated_demo",
                        "data_category": DATA_CAT_SIMULATED,
                        "reason": reason,
                    },
                )
        return out

    def _fetch_direct_tme(self, lookback_days: int = 10, max_pages: int = 4) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        collected: List[Dict[str, Any]] = []
        for ch in self.PUBLIC_HK_CHANNELS:
            before_id = None
            for _ in range(max_pages):
                url = f"https://t.me/s/{ch}" + (f"?before={before_id}" if before_id else "")
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    html_txt = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="replace")
                    blocks = html_txt.split("tgme_widget_message_wrap")
                    page_min_id = None
                    oldest = datetime.now(timezone.utc)
                    for blk in blocks[1:]:
                        txt_m = re.search(r"tgme_widget_message_text[^>]*>(.*?)</div>", blk, re.S)
                        dt_m = re.search(r'datetime="([^"]+)"', blk)
                        url_m = re.search(r'tgme_widget_message_date"[^>]*href="([^"]+)"', blk)
                        views_m = re.search(r'tgme_widget_message_views">([^<]+)<', blk)
                        if not txt_m:
                            continue
                        clean = unescape(re.sub(r"<[^>]+>", " ", txt_m.group(1))).strip()
                        post_url = url_m.group(1) if url_m else url
                        msg_id_str = post_url.rstrip("/").split("/")[-1]
                        if msg_id_str.isdigit():
                            mid = int(msg_id_str)
                            page_min_id = mid if page_min_id is None else min(page_min_id, mid)
                        date_str = dt_m.group(1) if dt_m else datetime.now(timezone.utc).isoformat()
                        try:
                            dt_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            if dt_obj < oldest:
                                oldest = dt_obj
                        except Exception:
                            pass
                        collected.append(
                            {
                                "id": msg_id_str,
                                "text": clean,
                                "date": date_str,
                                "viewsText": views_m.group(1).strip() if views_m else "0",
                                "url": post_url,
                                "channel": ch,
                            }
                        )
                    if page_min_id is None or page_min_id <= 1 or oldest < cutoff:
                        break
                    before_id = page_min_id
                except Exception:
                    break
        return collected

    def collect(
        self,
        stock: StockConfig,
        lookback_days: int = 10,
        max_items: int = 30,
    ) -> Tuple[List[NormalizedSocialPost], Dict[str, str]]:
        return self.collect_all([stock], lookback_days=lookback_days, max_items_per_stock=max_items)[stock.ticker]


def _parse_views(raw: str) -> int:
    s = raw.strip().upper()
    try:
        if s.endswith("K"):
            return int(float(s[:-1]) * 1000)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        return int(float(s))
    except ValueError:
        return 0
