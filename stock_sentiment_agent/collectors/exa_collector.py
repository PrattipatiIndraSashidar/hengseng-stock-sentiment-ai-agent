"""Official Exa Search API collector for financial news."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from stock_sentiment_agent.collectors.apify_collector import DATA_CAT_LIVE, DATA_CAT_SIMULATED
from stock_sentiment_agent.config import StockConfig, get_settings

FINANCIAL_KEYWORDS = [
    "earnings",
    "revenue",
    "financial results",
    "market trends",
    "investments",
    "product announcements",
    "regulatory",
    "analyst forecasts",
]


@dataclass
class NormalizedNewsArticle:
    ticker: str
    company_name: str
    headline: str
    article_url: str
    published_at: str
    published_utc: str
    summary: str
    source: str
    relevance_score: float
    is_simulated: bool = False
    data_category: str = DATA_CAT_LIVE


class ExaNewsCollector:
    """Collects recent financial news via Exa Search API with retries, deduplication, and relevance filtering."""

    EXA_SEARCH_URL = "https://api.exa.ai/search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = (api_key if api_key is not None else settings.exa_api_key).strip()
        self.sample_path = settings.data_dir / "sample_dataset.json"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> Tuple[bool, str]:
        """Verify EXA_API_KEY and probe the official Exa Search API endpoint."""
        if not self.is_configured:
            return False, "EXA_API_KEY is not configured; Exa Search API unavailable."
        _, err = self._post_exa_rest(
            {"query": "Tencent financial results", "type": "auto", "category": "news", "numResults": 1},
            max_retries=1,
        )
        if err:
            return False, err
        return True, "Exa Search API authenticated and responding."

    def collect(
        self,
        stock: StockConfig,
        lookback_days: int = 10,
        num_results: int = 8,
    ) -> Tuple[List[NormalizedNewsArticle], Dict[str, str]]:
        if not self.is_configured:
            demo = self._load_simulated_news(stock)
            return demo, {
                "status": "unavailable",
                "mode": "simulated_demo",
                "data_category": DATA_CAT_SIMULATED,
                "reason": "EXA_API_KEY not configured; Exa Search API unavailable. Using labeled SIMULATED DEMO DATA.",
            }

        start_dt = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = (
            f"{stock.company_name} ({stock.ticker}) financial results earnings revenue "
            "market trends investments product announcements regulatory news analyst forecasts"
        )
        payload = {
            "query": query,
            "type": "auto",
            "category": "news",
            "numResults": num_results,
            "startPublishedDate": start_dt,
            "contents": {
                "text": {"maxCharacters": 1200},
                "highlights": {"numSentences": 3},
            },
        }

        raw_results: List[Dict[str, Any]] = []
        error_msg: Optional[str] = None

        try:
            from exa_py import Exa  # type: ignore

            exa = Exa(self.api_key)
            resp = exa.search_and_contents(
                query,
                type="auto",
                category="news",
                num_results=num_results,
                start_published_date=start_dt,
                text={"max_characters": 1200},
                highlights={"num_sentences": 3},
            )
            for r in getattr(resp, "results", []):
                raw_results.append(
                    {
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "publishedDate": getattr(r, "published_date", ""),
                        "text": getattr(r, "text", ""),
                        "highlights": getattr(r, "highlights", []),
                        "score": getattr(r, "score", 0.7),
                    }
                )
        except ImportError:
            raw_results, error_msg = self._post_exa_rest(payload)
        except Exception as exc:
            error_msg = f"Exa SDK error: {exc}"

        if error_msg or not raw_results:
            demo = self._load_simulated_news(stock)
            return demo, {
                "status": "unavailable",
                "mode": "simulated_demo",
                "data_category": DATA_CAT_SIMULATED,
                "reason": f"Exa Search API failed ({error_msg or 'empty results'}); loaded labeled SIMULATED DEMO DATA.",
            }

        articles = self._normalize_and_filter(stock, raw_results)
        if not articles:
            demo = self._load_simulated_news(stock)
            return demo, {
                "status": "unavailable",
                "mode": "simulated_demo",
                "data_category": DATA_CAT_SIMULATED,
                "reason": "Exa returned no relevant articles after filtering; loaded labeled SIMULATED DEMO DATA.",
            }
        return articles, {
            "status": "connected",
            "mode": "live_exa",
            "data_category": DATA_CAT_LIVE,
            "reason": f"Collected {len(articles)} live financial news articles via Exa.",
        }

    def _post_exa_rest(self, payload: Dict[str, Any], max_retries: int = 2) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "HengSengSentimentAgent/1.0",
            "x-api-key": self.api_key,
        }
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(self.EXA_SEARCH_URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("results", []), None
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    return [], f"Exa API authentication failed (HTTP {exc.code})"
                if exc.code == 402:
                    return [], "Exa API credit/quota exceeded (HTTP 402)"
                if exc.code == 429 and attempt < max_retries:
                    time.sleep(float(exc.headers.get("Retry-After", 1.5 * (attempt + 1))))
                    continue
                if exc.code >= 500 and attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return [], f"Exa API HTTP {exc.code}: {exc.reason}"
            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                return [], f"Exa API request failed: {exc}"
        return [], "Exa API max retries exceeded"

    @staticmethod
    def _normalize_and_filter(
        stock: StockConfig, raw_results: List[Dict[str, Any]]
    ) -> List[NormalizedNewsArticle]:
        seen_urls = set()
        seen_titles = set()
        results: List[NormalizedNewsArticle] = []
        terms = [stock.company_name.lower(), stock.ticker.lower(), stock.ticker.split(".")[0].lower()] + [
            a.lower() for a in stock.aliases
        ]

        for item in raw_results:
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not url or not title or url in seen_urls or title.lower() in seen_titles:
                continue

            highlights = item.get("highlights") or []
            summary = " ".join(highlights) if isinstance(highlights, list) and highlights else str(item.get("text") or "")
            summary = summary.strip()[:1000]
            combined_lower = f"{title} {summary}".lower()

            entity_hits = sum(1 for t in terms if t in combined_lower)
            kw_hits = sum(1 for kw in FINANCIAL_KEYWORDS if kw in combined_lower)
            api_score = float(item.get("score") or 0.65)
            relevance = min(
                1.0,
                round(0.45 * min(1.0, api_score) + 0.35 * min(1.0, entity_hits / 2) + 0.20 * min(1.0, kw_hits / 3), 4),
            )
            if entity_hits == 0 and relevance < 0.45:
                continue

            pub = str(item.get("publishedDate") or datetime.now(timezone.utc).isoformat())
            source_domain = urlparse(url).netloc.replace("www.", "") or "financial-news"
            seen_urls.add(url)
            seen_titles.add(title.lower())
            results.append(
                NormalizedNewsArticle(
                    ticker=stock.ticker,
                    company_name=stock.company_name,
                    headline=title,
                    article_url=url,
                    published_at=pub,
                    published_utc=pub,
                    summary=summary or title,
                    source=source_domain,
                    relevance_score=relevance,
                    is_simulated=False,
                    data_category=DATA_CAT_LIVE,
                )
            )
        return results

    def _load_simulated_news(self, stock: StockConfig) -> List[NormalizedNewsArticle]:
        if not self.sample_path.exists():
            return []
        try:
            dataset = json.loads(self.sample_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: List[NormalizedNewsArticle] = []
        seen = set()
        for row in dataset.get("news_articles", []):
            if row.get("ticker") != stock.ticker:
                continue
            url = str(row.get("article_url") or "")
            if url in seen:
                continue
            seen.add(url)
            out.append(
                NormalizedNewsArticle(
                    ticker=stock.ticker,
                    company_name=stock.company_name,
                    headline=str(row["headline"]),
                    article_url=url,
                    published_at=str(row.get("published_at") or ""),
                    published_utc=str(row.get("published_utc") or ""),
                    summary=str(row.get("summary") or ""),
                    source=str(row.get("source") or "sample"),
                    relevance_score=float(row.get("relevance_score") or 0.8),
                    is_simulated=True,
                    data_category=DATA_CAT_SIMULATED,
                )
            )
        return out
