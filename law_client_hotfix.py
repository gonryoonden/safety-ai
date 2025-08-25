# law_client_hotfix.py
import os, time, logging
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

def _ensure_dirs(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _dump_text(path: str, text: str) -> None:
    _ensure_dirs(path)
    Path(path).write_text(text, encoding="utf-8", errors="ignore")

def _get_json_with_retry(session_get: Callable[[], requests.Response],
                         *,
                         dump_path: Optional[str] = None,
                         max_attempts: int = 3) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            resp = session_get()
            dt = int((time.time() - t0) * 1000)
            ct = (resp.headers.get("content-type") or "").lower()
            logger.info("GET %s %dms %s %s", getattr(resp, "url", ""), dt, resp.status_code, ct)
            resp.raise_for_status()
            text = resp.text

            if "json" not in ct:
                if dump_path:
                    _dump_text(dump_path, text)
                raise ValueError(f"Non-JSON body (ct={ct})")

            try:
                data = resp.json()
            except Exception as je:
                if dump_path:
                    _dump_text(dump_path, text)
                raise ValueError(f"JSON decode error: {je}") from je

            if dump_path:
                _dump_text(dump_path, text)
            return data

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            logger.warning("Attempt %d/%d failed (timeout/conn): %s", attempt, max_attempts, e)
            time.sleep(min(2.0, 0.5 * attempt))
        except Exception as e:
            last_err = e
            logger.warning("Attempt %d/%d failed: %s", attempt, max_attempts, e)
            time.sleep(min(2.0, 0.5 * attempt))
    assert last_err is not None
    raise last_err


class LawAPIClient:
    """
    - 모든 요청에 타임아웃 강제 (기본: connect 5s, read 15s)
    - 429/5xx 재시도
    - 응답이 JSON이 아니면 바로 덤프 후 실패
    """
    def __init__(self, oc: str, base: str = None, timeout: Optional[tuple] = None):
        self.base = (base or os.environ.get("LAW_BASE") or "http://www.law.go.kr").rstrip("/")
        self.oc = oc
        # (connect, read) 튜플. None이면 기본값 사용
        self.timeout = timeout or (float(os.getenv("LAW_CONNECT_TIMEOUT", "5")),
                                   float(os.getenv("LAW_READ_TIMEOUT", "15")))
        s = requests.Session()
        retries = Retry(
            total=3, connect=3, read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json,text/plain,*/*",
        })
        self.session = s

    def _get(self, path: str, params: Dict[str, Any]) -> requests.Response:
        url = f"{self.base}{path}"
        # 요청마다 타임아웃 확실히 적용
        return self.session.get(url, params=params, timeout=self.timeout)

    def get_law(self, mst: str, jo: Optional[str] = None) -> Dict[str, Any]:
        p = {"OC": self.oc, "target": "law", "type": "JSON", "MST": mst}
        if jo:
            p["JO"] = jo
            dump = f"laws/raw/mst_{mst}_jo_{jo}.json"
        else:
            dump = f"laws/raw/mst_{mst}_all.json"
        return _get_json_with_retry(lambda: self._get("/DRF/lawService.do", p),
                                    dump_path=dump,
                                    max_attempts=3)
