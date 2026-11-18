"""Base Apify integration with token verification, actor checks, pagination, rate-limit backoff, and deduplication."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.preprocessing.data_cleaning import text_fingerprint

DATA_CAT_LIVE = "LIVE DATA"
DATA_CAT_HISTORICAL = "HISTORICAL DATA"
DATA_CAT_SIMULATED = "SIMULATED DEMO DATA"


@dataclass
class NormalizedSocialPost:
    ticker: str
    company_name: str
    post_id: str
    post_text: str
    post_timestamp: str
    utc_timestamp: str
    source_platform: str
    author_username: Optional[str]
    author_follower_count: Optional[int]
    likes: int
    comments: int
    reposts: int
    views: int
    post_url: Optional[str]
    raw_payload: str
    is_simulated: bool = False
    data_category: str = DATA_CAT_LIVE


class ApifyBaseCollector:
    """Shared Apify actor runner supporting apify-client or direct Apify v2 REST API."""

    BASE_URL = "https://api.apify.com/v2"

    def __init__(self, api_token: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_token = (api_token if api_token is not None else settings.apify_api_token).strip()
        self.sample_path = settings.data_dir / "sample_dataset.json"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_token)

    def _http_json(
        self,
        url: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        timeout: int = 20,
        max_retries: int = 2,
    ) -> Tuple[Optional[Any], Optional[str], int]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "HengSengSentimentAgent/1.0",
        }
        data_bytes = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data_bytes = json.dumps(body).encode("utf-8")

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else {}, None, resp.status
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    raw_err = exc.read().decode("utf-8", errors="replace")
                    parsed_err = json.loads(raw_err)
                    msg = parsed_err.get("error", {}).get("message")
                    if msg:
                        detail = f" - {str(msg)[:180]}"
                except Exception:
                    pass
                if exc.code == 429 and attempt < max_retries:
                    retry_after = float(exc.headers.get("Retry-After", 1.5 * (attempt + 1)))
                    time.sleep(min(retry_after, 5.0))
                    continue
                if exc.code >= 500 and attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None, f"HTTP {exc.code}: {exc.reason}{detail}", exc.code
            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                return None, str(exc), 0
        return None, "Max retries exceeded", 0

    def verify_authentication(self) -> Tuple[bool, str]:
        if not self.is_configured:
            return False, "APIFY_API_TOKEN is not configured."
        payload, err, status = self._http_json(f"{self.BASE_URL}/users/me?token={self.api_token}", timeout=8, max_retries=1)
        if status in (401, 403):
            return False, "APIFY_API_TOKEN authentication rejected (HTTP 401/403)."
        if err:
            return False, f"Apify API verification failed: {err}"
        username = (payload or {}).get("data", {}).get("username", "authenticated")
        return True, f"Authenticated as Apify user '{username}'."

    def verify_actor_available(self, actor_id: str) -> Tuple[bool, str]:
        if not self.is_configured:
            return False, "APIFY_API_TOKEN is not configured."
        actor_slug = actor_id.replace("/", "~")
        payload, err, status = self._http_json(
            f"{self.BASE_URL}/acts/{actor_slug}?token={self.api_token}", timeout=8, max_retries=1
        )
        if status == 404:
            return False, f"Apify actor '{actor_id}' was not found (HTTP 404)."
        if status in (401, 403):
            return False, f"APIFY_API_TOKEN unauthorized for actor '{actor_id}'."
        if err or not payload:
            return False, f"Unable to verify actor '{actor_id}': {err}"
        return True, f"Actor '{actor_id}' verified."

    def probe_sources(self) -> Dict[str, Dict[str, Any]]:
        """Verify APIFY_API_TOKEN auth and each regional actor without fabricating success."""
        actor_map = {
            "LIHKG": ("HKEX", "apify/cheerio-scraper"),
            "Telegram": ("HKEX", "apify/cheerio-scraper"),
            "Naver Finance": ("KRX", "apify/cheerio-scraper"),
            "Daum Finance": ("KRX", "apify/web-scraper"),
        }
        if not self.is_configured:
            return {
                plat: {
                    "platform": plat,
                    "exchange": exch,
                    "actor_id": actor,
                    "status": "unavailable",
                    "token_configured": False,
                    "actor_verified": False,
                    "reason": "APIFY_API_TOKEN is not configured; cannot authenticate to Apify.",
                }
                for plat, (exch, actor) in actor_map.items()
            }

        auth_ok, auth_msg = self.verify_authentication()
        actor_cache: Dict[str, Tuple[bool, str]] = {}
        report: Dict[str, Dict[str, Any]] = {}
        for plat, (exch, actor) in actor_map.items():
            if not auth_ok:
                report[plat] = {
                    "platform": plat,
                    "exchange": exch,
                    "actor_id": actor,
                    "status": "unavailable",
                    "token_configured": True,
                    "actor_verified": False,
                    "reason": auth_msg,
                }
                continue
            if actor not in actor_cache:
                actor_cache[actor] = self.verify_actor_available(actor)
            act_ok, act_msg = actor_cache[actor]
            report[plat] = {
                "platform": plat,
                "exchange": exch,
                "actor_id": actor,
                "status": "connected" if act_ok else "unavailable",
                "token_configured": True,
                "actor_verified": act_ok,
                "reason": act_msg if not act_ok else f"{auth_msg} {act_msg}",
            }
        return report

    def run_actor_and_fetch_items(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
        max_items: int = 100,
        page_size: int = 50,
        timeout_sec: int = 60,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Verify actor availability, execute run, and paginate dataset items with rate-limit handling."""
        if not self.is_configured:
            return [], "APIFY_API_TOKEN is missing."

        actor_ok, actor_msg = self.verify_actor_available(actor_id)
        if not actor_ok:
            return [], actor_msg

        # Try official apify-client if installed
        try:
            from apify_client import ApifyClient  # type: ignore

            client = ApifyClient(self.api_token)
            run = client.actor(actor_id).call(run_input=run_input, timeout_secs=timeout_sec)
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                return [], f"Apify actor {actor_id} returned no dataset ID."

            items: List[Dict[str, Any]] = []
            offset = 0
            while len(items) < max_items:
                page = client.dataset(dataset_id).list_items(offset=offset, limit=page_size)
                batch = page.items or []
                if not batch:
                    break
                items.extend(batch)
                offset += len(batch)
                if len(batch) < page_size:
                    break
            return items[:max_items], None
        except ImportError:
            pass
        except Exception as exc:
            return [], f"Apify actor {actor_id} execution failed: {exc}"

        # Direct Apify REST API v2 with run polling and dataset pagination
        actor_slug = actor_id.replace("/", "~")
        run_resp, run_err, _ = self._http_json(
            f"{self.BASE_URL}/acts/{actor_slug}/runs?token={self.api_token}&waitForFinish={min(timeout_sec, 45)}",
            method="POST",
            body=run_input,
            timeout=min(timeout_sec, 50),
        )
        if run_err or not isinstance(run_resp, dict):
            return [], f"Apify run start failed for {actor_id}: {run_err}"

        run_data = run_resp.get("data", {})
        dataset_id = run_data.get("defaultDatasetId")
        run_status = run_data.get("status", "UNKNOWN")
        if run_status not in ("SUCCEEDED", "RUNNING") and not dataset_id:
            return [], f"Apify actor {actor_id} ended with status {run_status}."

        if not dataset_id:
            return [], f"Apify actor {actor_id} produced no dataset ID."

        items = []
        offset = 0
        while len(items) < max_items:
            limit = min(page_size, max_items - len(items))
            page_url = (
                f"{self.BASE_URL}/datasets/{dataset_id}/items"
                f"?token={self.api_token}&offset={offset}&limit={limit}&clean=true"
            )
            batch, page_err, _ = self._http_json(page_url, timeout=15)
            if page_err:
                return items, f"Dataset pagination error at offset {offset}: {page_err}"
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            offset += len(batch)
            if len(batch) < limit:
                break
        return items[:max_items], None

    @staticmethod
    def deduplicate_posts(posts: List[NormalizedSocialPost]) -> List[NormalizedSocialPost]:
        seen_ids = set()
        seen_hashes = set()
        unique: List[NormalizedSocialPost] = []
        for post in posts:
            text = post.post_text.strip()
            if not text:
                continue
            id_key = (post.source_platform, post.post_id)
            text_key = (post.ticker, post.source_platform, text_fingerprint(text))
            if id_key not in seen_ids and text_key not in seen_hashes:
                seen_ids.add(id_key)
                seen_hashes.add(text_key)
                unique.append(post)
        return unique

    def load_simulated_posts(
        self,
        stock: StockConfig,
        platform: str,
        lookback_days: int = 10,
    ) -> List[NormalizedSocialPost]:
        """Load clearly labeled SIMULATED DEMO DATA when live Apify credentials or platform access are unavailable."""
        if not self.sample_path.exists():
            return []
        try:
            dataset = json.loads(self.sample_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        results: List[NormalizedSocialPost] = []
        for row in dataset.get("social_posts", []):
            if row.get("ticker") != stock.ticker or row.get("source_platform") != platform:
                continue
            utc_str = str(row.get("utc_timestamp", ""))
            try:
                dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                if dt < cutoff - timedelta(days=45):
                    pass
            except Exception:
                pass
            results.append(
                NormalizedSocialPost(
                    ticker=stock.ticker,
                    company_name=stock.company_name,
                    post_id=str(row["post_id"]),
                    post_text=str(row["post_text"]),
                    post_timestamp=str(row.get("post_timestamp", utc_str)),
                    utc_timestamp=utc_str,
                    source_platform=platform,
                    author_username=row.get("author_username"),
                    author_follower_count=row.get("author_follower_count"),
                    likes=int(row.get("likes") or 0),
                    comments=int(row.get("comments") or 0),
                    reposts=int(row.get("reposts") or 0),
                    views=int(row.get("views") or 0),
                    post_url=row.get("post_url"),
                    raw_payload=json.dumps(
                        {**row, "_data_category": DATA_CAT_SIMULATED, "_simulated_demo_mode": True},
                        ensure_ascii=False,
                    ),
                    is_simulated=True,
                    data_category=DATA_CAT_SIMULATED,
                )
            )
        return self.deduplicate_posts(results)

    @staticmethod
    def save_raw_snapshot(platform: str, ticker: str, rows: List[NormalizedSocialPost], data_dir: Path) -> None:
        raw_dir = data_dir / "raw" / "social"
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe_ticker = ticker.replace(".", "_")
        out_path = raw_dir / f"{platform.lower().replace(' ', '_')}_{safe_ticker}.json"
        out_path.write_text(
            json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
