import os
from dotenv import load_dotenv
load_dotenv()
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InsecureRequestWarning # 👈 1. 이 줄 추가

requests.packages.urllib3.disable_warnings(InsecureRequestWarning) # 👈 2. 이 줄 추가

LAW_BASE = os.environ.get("LAW_BASE", "http://www.law.go.kr")

logger = logging.getLogger(__name__)

def _parse_response_as_json(resp: requests.Response) -> Dict[str, Any]:
    ct = (resp.headers.get("Content-Type") or "").lower()
    body = resp.text or ""
    if "application/json" in ct:
        try:
            return resp.json()
        except Exception as e:
            logger.error("JSON decode failed ct=%s len=%d preview=%r", ct, len(body), body[:200])
            raise
    if not body.strip():
        logger.error("Empty body with 200 OK ct=%s", ct)
        raise RuntimeError("Empty 200 response")
    if body.lstrip().startswith("<"):  # HTML/XML 가능
        logger.error("Non-JSON body ct=%s len=%d preview=%r", ct, len(body), body[:200])
        raise RuntimeError("Non-JSON (HTML/XML) response")
    logger.error("Unexpected content-type ct=%s preview=%r", ct, body[:200])
    raise RuntimeError("Unexpected non-JSON response")

def _get_json_with_retry(session_get):
    # 호출부에서 lambda로 주입 or 간단히 _get 안에서 사용
    for i in range(3):  # 0,1,2
        try:
            resp = session_get()
            return _parse_response_as_json(resp)
        except Exception as e:
            if i == 2:
                raise
            time.sleep(1.0 * (2 ** i))


class RateLimiter:
    """Simple token bucket rate limiter for QPS across process.

    - capacity: max tokens
    - refill_rate: tokens per second
    """

    def __init__(self, max_rps: int = 5):
        self.capacity = max(1, int(max_rps))
        self.tokens = float(self.capacity)
        self.refill_rate = float(self.capacity)  # tokens per second
        self.last_refill = time.monotonic()
        self.cv = threading.Condition()

    def acquire(self):
        with self.cv:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                    self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # Not enough tokens, wait a bit
                wait_time = max(0.001, (1.0 - self.tokens) / self.refill_rate)
                logger.debug("Throttling: sleeping %.3fs", wait_time)
                self.cv.wait(timeout=wait_time)


class LawAPIClient:
    def __init__(self,
                 oc: Optional[str] = None,
                 max_rps: Optional[int] = None,
                 timeout: Optional[int] = None):
        self.oc = oc or os.environ.get("LAW_API_OC", "test")
        self.timeout = float(timeout or os.environ.get("REQUEST_TIMEOUT", 15))
        self.rate_limiter = RateLimiter(max_rps=int(max_rps or os.environ.get("MAX_RPS", 5)))
        self.session = self._build_session()
        self.session.verify = False  # 👈 이 줄을 추가하세요.

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({
        # 브라우저급 UA로 통일
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        # JSON만 수락 (굳이 HTML 선호 주지 않기)
        "Accept": "application/json",
        # 선택: 한국어 환경과 동일하게
        "Accept-Language": "ko-KR,ko;q=0.9"
        })
        return s

    def _get(self, path: str, params: Dict[str, Any]) -> requests.Response:
        self.rate_limiter.acquire()
        url = LAW_BASE + path
        p = {"OC": self.oc, **params}
        t0 = time.monotonic()
        resp = self.session.get(url, params=p, timeout=self.timeout)
        dt = time.monotonic() - t0
        logger.info("GET %s %.0fms %s ct=%s url=%s",
                path, dt * 1000, resp.status_code,
                resp.headers.get("Content-Type"), resp.url)        
        resp.raise_for_status()
        return resp

    # 검색(목록)
    def search_law(self, query: str, page: int = 1, display: int = 20, sort: str = "efdes") -> Dict[str, Any]:
        params = {
            "target": "law",
            "type": "JSON",
            "query": query,
            "page": page,
            "display": display,
            "sort": sort,
        }
        # 반드시 재시도 포함 버전으로 교체
        return _get_json_with_retry(lambda: self._get("/DRF/lawSearch.do", params))

    # 본문
    def get_law(self, mst: str, jo: Optional[str] = None) -> Dict[str, Any]:
        params = {"target": "law", "type": "JSON", "MST": mst}
        if jo:
            params["JO"] = jo.zfill(6)  # 6자리 패딩 권장
        return _get_json_with_retry(lambda: self._get("/DRF/lawService.do", params))

    # 별표/서식
    def search_attachments(
        self,
        query: str,
        page: int = 1,
        display: int = 50,
        sort: str = "lasc",
        org: Optional[str] = None,
        knd: Optional[str] = None,
        search: Optional[int] = None,  # 1:별표/서식명, 2:해당법령검색, 3:별표본문검색
    ) -> Dict[str, Any]:
        params = {
            "target": "licbyl",
            "type": "JSON",
            "query": query or "*",
            "page": page,
            "display": display,
            "sort": sort,
        }
        if org:
            params["org"] = org
        if knd:
            params["knd"] = knd
        if search is not None:
            params["search"] = int(search)

        return _get_json_with_retry(lambda: self._get("/DRF/lawSearch.do", params))

def _safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _normalize_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _parse_pdf_link_from_law_json(law_json: Dict[str, Any], attachment_number: int) -> Optional[str]:
    # Best-effort parse: look for fields with names like '별표' and attachment indexes
    try:
        law = _safe_get(law_json, "법령") or _safe_get(law_json, "Law") or law_json
        # Some responses embed attachment list under '별표목록' or similar
        candidates = []
        for k, v in (law.items() if isinstance(law, dict) else []):
            if isinstance(v, (list, dict)) and ("별표" in k or "서식" in k or "부록" in k):
                candidates.extend(_normalize_list(v))
        # Fall back: search all values for dicts with fields '별표번호' and '별표서식PDF파일링크'
        def walk(obj):
            results = []
            if isinstance(obj, dict):
                if "별표번호" in obj or "별표서식PDF파일링크" in obj:
                    results.append(obj)
                for vv in obj.values():
                    results.extend(walk(vv))
            elif isinstance(obj, list):
                for it in obj:
                    results.extend(walk(it))
            return results
        if not candidates:
            candidates = walk(law)
        for item in candidates:
            try:
                num = _safe_get(item, "별표번호") or _safe_get(item, "번호")
                if num is None:
                    continue
                # Some numbers can be zero-padded strings
                try:
                    n = int(str(num).lstrip("0") or "0")
                except Exception:
                    continue
                if n == int(attachment_number):
                    pdf = _safe_get(item, "별표서식PDF파일링크") or _safe_get(item, "pdfLink")
                    if pdf:
                        if pdf.startswith("/"):
                            return LAW_BASE + pdf
                        return pdf
            except Exception:
                continue
    except Exception:
        return None
    return None


def map_law_name_to_mst(client: LawAPIClient, law_name: str) -> Optional[str]:
    """Map law_name to latest MST using lawSearch API.
    Choose the item with latest 시행일자/공포일자.
    """
    data = client.search_law(query=law_name, display=50, sort="efdes")
    law_list = _safe_get(_safe_get(data, "LawSearch"), "law")
    items = _normalize_list(law_list)
    if not items:
        return None
    def keyf(it: Dict[str, Any]) -> Tuple[str, str]:
        return (
            _safe_get(it, "시행일자") or "",
            _safe_get(it, "공포일자") or "",
        )
    latest = sorted(items, key=keyf, reverse=True)[0]
    mst = _safe_get(latest, "법령일련번호") or _safe_get(latest, "MST")
    return str(mst) if mst is not None else None


def get_attachment_link(law_name: str, attachment_number: int,
                        client: Optional[LawAPIClient] = None) -> Dict[str, Any]:
    """Return attachment PDF URL by law name and attachment number.

    Strategy:
      1) Try lawService by MST (resolved via lawSearch) and parse attachment PDF link
      2) Fallback: licbyl search by 관련법령명 + 별표번호
      3) Fallback: return law detail page link with message
    """
    client = client or LawAPIClient()
    result: Dict[str, Any] = {"ok": False, "law_name": law_name, "attachment_number": attachment_number}
    try:
        mst = map_law_name_to_mst(client, law_name)
        if mst:
            law_json = client.get_law(mst)
            pdf = _parse_pdf_link_from_law_json(law_json, attachment_number)
            if pdf:
                result.update({"ok": True, "pdf_url": pdf, "mst": mst, "source": "lawService"})
                return result
        # Plan B: licbyl search
        data = client.search_attachments(query=law_name, display=100)
        lic = _safe_get(_safe_get(data, "licBylSearch"), "licbyl")
        for item in _normalize_list(lic):
            try:
                law = _safe_get(item, "관련법령명") or _safe_get(item, "lawName")
                num = _safe_get(item, "별표번호")
                if law and law_name in str(law):
                    n = int(str(num).lstrip("0") or "0") if num is not None else None
                    if n == int(attachment_number):
                        pdf = _safe_get(item, "별표서식PDF파일링크")
                        if pdf:
                            url = LAW_BASE + pdf if str(pdf).startswith("/") else str(pdf)
                            result.update({"ok": True, "pdf_url": url, "source": "licbyl"})
                            # Add MST if available
                            if mst:
                                result["mst"] = mst
                            return result
            except Exception:
                continue
        # Plan C: Provide law detail page
        if mst:
            detail = f"{LAW_BASE}/DRF/lawService.do?OC={client.oc}&target=law&MST={mst}&type=HTML"
            result.update({
                "ok": False,
                "detail_page": detail,
                "mst": mst,
                "message": "PDF link not found. Provided law detail page instead."
            })
        else:
            result.update({
                "ok": False,
                "message": "Unable to resolve law name to MST; please refine the law name."
            })
        return result
    except Exception as e:
        logger.exception("get_attachment_link failed: %s", e)
        result.update({"ok": False, "error": str(e)})
        return result


__all__ = [
    "LawAPIClient",
    "RateLimiter",
    "map_law_name_to_mst",
    "get_attachment_link",
]

if __name__ == "__main__":
    print("LAW_API_OC =", os.environ.get("LAW_API_OC"))



# --- Law name resolution / lightweight meta ---


_DEF_TITLE_KEYS = (
    "\ubc95\ub839\uba85\ud55c\uae00",
    "\ubc95\ub839\uba85",
    "\ubc95\ub839\uc57d\uce6d",
)
_DEF_MST_KEYS = (
    "\ubc95\ub839\uc77c\ub828\ubc88\ud638",
    "\ubc95\ub839ID",
    "MST",
)


def _normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    t = str(s).strip()
    for ch in (" ", "\u00b7", "\u318d", "\t", "\n", "\r", "(", ")"):
        t = t.replace(ch, "")
    return t


def _extract_law_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    law_list = _safe_get(_safe_get(data, "LawSearch"), "law")
    return _normalize_list(law_list)


def _pick_field_by_substr(item: Dict[str, Any], substrs: Tuple[str, ...]) -> Optional[str]:
    for k, v in item.items():
        if any(s in str(k) for s in substrs):
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def _pick_field_exact(item: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _date_to_int(s: Optional[str]) -> int:
    if not s:
        return 0
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    return int(digits) if digits else 0


def _to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    return int(digits) if digits else None


def resolve_law_name(client: LawAPIClient, law_name: str) -> Dict[str, Any]:
    data = client.search_law(query=law_name, display=50, sort="efdes")
    items = _extract_law_items(data)
    norm_query = _normalize_name(law_name)

    candidates = []
    for it in items:
        title = _pick_field_exact(it, _DEF_TITLE_KEYS) or _pick_field_by_substr(
            it, ("\ubc95\ub839\uba85",)
        )
        mst = _pick_field_exact(it, _DEF_MST_KEYS) or _pick_field_by_substr(
            it, ("\ubc95\ub839\uc77c\ub828\ubc88\ud638", "\ubc95\ub839ID", "MST")
        )
        law_type = _pick_field_by_substr(
            it, ("\ubc95\ub839\uad6c\ubd84", "\ubc95\ub839\uc885\ub958", "\ubc95\ub839\uad6c\ubd84\uba85")
        )
        effective = _pick_field_by_substr(it, ("\uc2dc\ud589", "\uc2dc\ud589\uc77c"))
        amended = _pick_field_by_substr(it, ("\uac1c\uc815", "\uac1c\uc815\uc77c"))
        promulgated = _pick_field_by_substr(it, ("\uacf5\ud3ec", "\uacf5\ud3ec\uc77c"))
        article_cnt = _pick_field_by_substr(it, ("\uc870\ubb38", "\uc870\ubb38\uc218"))
        annex_cnt = _pick_field_by_substr(it, ("\ubcc4\ud45c", "\ubcc4\ud45c\uc218"))

        if not title and not mst:
            continue

        score = 0
        norm_title = _normalize_name(title)
        if norm_title and norm_title == norm_query:
            score += 120
        elif norm_title and (norm_query in norm_title or norm_title in norm_query):
            score += 70
        if law_type and ("\ubc95\ub839" in law_type):
            score += 20
        if law_type and ("\ubc95\ub960" in law_type):
            score += 10

        candidates.append(
            {
                "title": title,
                "mst": mst,
                "law_type": law_type,
                "effective_date": effective,
                "latest_amended": amended,
                "promulgated_date": promulgated,
                "article_count": _to_int(article_cnt),
                "annex_count": _to_int(annex_cnt),
                "score": score,
                "sort_date": max(
                    _date_to_int(effective),
                    _date_to_int(amended),
                    _date_to_int(promulgated),
                ),
            }
        )

    if not candidates:
        return {
            "status": "unresolved",
            "law_name": law_name,
            "reason": "no_candidates",
        }

    candidates.sort(key=lambda x: (x["score"], x["sort_date"]), reverse=True)
    best = candidates[0]
    if not best.get("mst"):
        return {
            "status": "unresolved",
            "law_name": law_name,
            "reason": "missing_mst",
        }

    if best["score"] >= 120:
        conf = "high"
    elif best["score"] >= 80:
        conf = "medium"
    elif best["score"] >= 60:
        conf = "low"
    else:
        conf = "low"

    return {
        "status": "resolved",
        "law_name": law_name,
        "mst": best.get("mst"),
        "resolved_title": best.get("title"),
        "match_confidence": conf,
        "effective_date": best.get("effective_date"),
        "latest_amended": best.get("latest_amended"),
        "promulgated_date": best.get("promulgated_date"),
        "article_count": best.get("article_count"),
        "annex_count": best.get("annex_count"),
        "law_type": best.get("law_type"),
    }


def fetch_law_meta_by_name(client: LawAPIClient, law_name: str) -> Dict[str, Any]:
    result = resolve_law_name(client, law_name)
    result["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


__all__ = [
    "LawAPIClient",
    "map_law_name_to_mst",
    "resolve_law_name",
    "fetch_law_meta_by_name",
]
