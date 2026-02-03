# vector_search_service.py
import os
import re
import json
import time
import logging
import tempfile
import unicodedata
import hashlib
import numpy as np
from urllib.parse import urljoin
from typing import Any, Dict, Iterable, List, Optional, Tuple
from normalizers import postprocess_units, _normalize_text
from io import BytesIO
from bs4 import BeautifulSoup  # pip install beautifulsoup4
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# --- Annex HTML/PDF parsing helpers (module-level) ---
def _clean_md(s: str) -> str:
    return "\n".join(line.rstrip() for line in (s or "").splitlines()).strip()

def _html_to_markdown(html: str) -> Tuple[str, List[Dict[str, Any]]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    blocks: List[str] = []
    tables_json: List[Dict[str, Any]] = []

    # 표 → 마크다운 + JSON
    for tbl in soup.find_all("table"):
        rows = []
        for tr in tbl.find_all("tr"):
            cols = [(" ".join(td.stripped_strings)) for td in tr.find_all(["th", "td"])]
            # 완전 빈 행은 스킵
            if cols and any(c.strip() for c in cols):
                rows.append(cols)
        if not rows:
            continue

        # 헤더 유무 판단: <th>가 있으면 그걸 헤더로, 없으면 첫 행이 데이터이니 가짜 헤더 생성
        has_th = bool(tbl.find("th"))
        if has_th:
            header = rows[0]
            data_rows = rows[1:]
        else:
            header = [f"열{i+1}" for i in range(len(rows[0]))]
            data_rows = rows

        # 마크다운 테이블 생성
        md = []
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in data_rows:
            md.append("| " + " | ".join(r) + " |")
        blocks.append("\n".join(md))

        tables_json.append({"headers": header, "rows": data_rows})
        tbl.decompose()

    # 본문 텍스트 추출
    text_parts = []
    for p in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","div","span"]):
        t = " ".join(p.stripped_strings)
        if t:
            text_parts.append(t)
    text_md = "\n".join(text_parts)

    joined = "\n\n".join(part for part in [text_md] + blocks if part.strip())
    return _clean_md(joined), tables_json

def _is_meaningful_annex(md: Optional[str], tables: Optional[List[Dict[str, Any]]]) -> bool:
    """
    본문이 헤더/레이블만인지 판정.
    - '별표·서식' 류 잡문구 제거 후 충분한 길이인지
    - 표에 데이터 셀이 실제로 있는지
    """
    text = (md or "")
    # 흔한 헤더/잡문구 제거(변형들 포함)
    junk = [
        "별표·서식", "별표 · 서식", "별표 ·서식", "별표· 서식",
        "첨부파일", "다운로드", "보기", "목록", "프린트", "인쇄",
    ]
    for j in junk:
        text = text.replace(j, "")
    # 공백/개행 정리
    text = "\n".join(line.strip() for line in text.splitlines())
    text = " ".join(text.split())
    text = text.strip()

    # 표 데이터 유무(헤더만 있고 데이터가 없으면 False)
    has_table_cells = False
    for t in (tables or []):
        rows = t.get("rows") or []
        for row in rows:
            if any((c or "").strip() for c in row):
                has_table_cells = True
                break
        if has_table_cells:
            break

    # 너무 짧은 안내문/헤더만 있으면 False
    # (실제 내용은 보통 100자 이상이거나 표 데이터가 존재)
    return (len(text) >= 60) or has_table_cells


def _fetch_annex_body_with_links(
    client: "LawAPIClient",
    links: Dict[str, str]
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    links: {"detail":..., "html":..., "pdf":...}
    우선순위: detail(HTML) → html(파일) → pdf
    return: (본문_MD, tables_json, source_tag)
    """
    global _ANNEX_PDF_CACHE_DIRTY

            # --- URL 보정: 상대경로를 절대경로로, detail은 모바일 뷰 우선 ---
    base = "https://www.law.go.kr"
    def _abs(u):
        return urljoin(base, u) if u else None

    links = {k: _abs(v) for k, v in (links or {}).items()}

    # detail의 모바일 뷰(iframe 없이 본문 노출)를 기본으로 강제
    detail_mobile = None
    if links.get("detail"):
        if "mobileYn=" in links["detail"]:
            detail_mobile = re.sub(r"mobileYn=[^&]*", "mobileYn=Y", links["detail"])
        else:
            sep = "&" if "?" in links["detail"] else "?"
            detail_mobile = links["detail"] + sep + "mobileYn=Y"
        links["detail"] = detail_mobile  # ← 이후 로직이 이 값으로 요청하게 바꿈


    
    # --- JSON(licbyl) 상세 폴백: detail 링크의 ID로 '...내용'을 받아오기 ---
    detail = links.get("detail") or ""   # 패치1에서 보정한 detail을 사용
    m = re.search(r"[?&]ID=([0-9]+)", detail)
    if m:
        lic_id = m.group(1)
        try:
            client.rate_limiter.acquire()
            urlj = f"https://www.law.go.kr/DRF/lawService.do?OC={client.oc}&target=licbyl&ID={lic_id}&type=JSON"
            respj = client.session.get(urlj, timeout=client.timeout, verify=False)
            j = respj.json()

            # JSON 어디에 있든 '*내용' 키를 찾아 HTML 본문으로 간주
            def _find_content(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if ("내용" in str(k)) and isinstance(v, str) and v.strip():
                            return v
                        got = _find_content(v)
                        if got: return got
                elif isinstance(x, list):
                    for v in x:
                        got = _find_content(v)
                        if got: return got
                return None

            html = _find_content(j)
            if html:
                md, tables = _html_to_markdown(html)
                md = _clean_md(md)
                if _is_meaningful_annex(md, tables):
                    return md, tables, "licbyl-json"
        except Exception:
            pass

   
    order = ["detail", "html", "pdf"]
    for key in order:
        url = (links or {}).get(key)
        if not url:
            continue
        try:
            client.rate_limiter.acquire()
            resp = client.session.get(url, timeout=client.timeout, verify=False)
            ct = (resp.headers.get("Content-Type") or "").lower()

            # ---- PDF/HTML 추가 판별자 ----
            cd = (resp.headers.get("Content-Disposition") or "")
            # Content-Disposition 파일명 추출 (확장자 판단용)
            m_fn = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.I)
            filename = (m_fn.group(1) if m_fn else "").strip()
            filename_l = filename.lower()

            is_pdf_by_header = ("pdf" in ct) or filename_l.endswith(".pdf")
            # 바이트 시그니처로도 PDF 판정 (flDownload.do는 종종 octet-stream으로 내려옴)
            content_head = resp.content[:5] if resp.content else b""
            is_pdf_by_magic = content_head.startswith(b"%PDF-")

            # HTML도 octet-stream으로 내려오는 경우가 있으니, 텍스트가 '<' 로 시작하면 HTML로 간주
            text_head = (resp.text or "").lstrip()[:1]
            is_html_loose = ("html" in ct) or text_head == "<"


            # HTML 계열 처리
            # HTML 계열 처리
            if key in ("detail", "html") and is_html_loose:
                html = resp.text

                # detail이 iframe 껍데기면 내부 본문 재요청
                try:
                    soup0 = BeautifulSoup(html, "html.parser")
                    frame = soup0.find("iframe", src=True)
                    if frame and frame.get("src"):
                        nested_url = urljoin(url, frame["src"])
                        resp2 = client.session.get(nested_url, timeout=client.timeout, verify=False)
                        ct2 = (resp2.headers.get("Content-Type") or "").lower()
                        if "html" in ct2 and resp2.text.strip():
                            html = resp2.text
                except Exception:
                    pass

                md, tables = _html_to_markdown(html)
                md = _clean_md(md)

                if not _is_meaningful_annex(md, tables):
                    # ⬇⬇ detail/html 페이지 안에 있는 실제 컨텐츠 링크를 한 번 더 찾아본다
                    try:
                        soup1 = BeautifulSoup(html, "html.parser")
                        cand_urls: List[str] = []

                        # 1) a[href] 후보들 수집 (flDownload/pdf 우선)
                        for a in soup1.select("a[href]"):
                            href = a.get("href")
                            if not href:
                                continue
                            href = href.strip()
                            if any(x in href.lower() for x in ("fldownload.do", ".pdf", "pdfdown", "filedown", "fldownload")):
                                cand_urls.append(urljoin(url, href))

                        # 2) object/embed/iframe도 후보로
                        for tag in soup1.find_all(["object", "embed", "iframe"]):
                            src = tag.get("data") or tag.get("src")
                            if not src:
                                continue
                            src = str(src).strip()
                            if any(x in src.lower() for x in (".pdf", "fldownload.do")):
                                cand_urls.append(urljoin(url, src))

                        # 중복 제거 + 최대 3개까지만 시도
                        seen = set()
                        cand_urls = [u for u in cand_urls if not (u in seen or seen.add(u))][:3]

                        for u2 in cand_urls:
                            try:
                                resp2 = client.session.get(u2, timeout=client.timeout, verify=False)
                                ct2 = (resp2.headers.get("Content-Type") or "").lower()
                                cd2 = (resp2.headers.get("Content-Disposition") or "")
                                m_fn2 = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd2, flags=re.I)
                                filename2 = (m_fn2.group(1) if m_fn2 else "").strip().lower()

                                # --- 느슨한 판별자: HTML/PDF 모두 header/내용/URL로 감지 ---
                                is_pdf_by_header2 = ("pdf" in ct2) or filename2.endswith(".pdf")
                                content_head2 = resp2.content[:5] if resp2.content else b""
                                is_pdf_by_magic2 = content_head2.startswith(b"%PDF-")
                                text_head2 = (resp2.text or "").lstrip()[:1]
                                is_html_loose2 = ("html" in ct2) or text_head2 == "<"

                                # HTML이면 다시 파싱
                                if is_html_loose2:
                                    md2, tables2 = _html_to_markdown(resp2.text)
                                    md2 = _clean_md(md2)
                                    if _is_meaningful_annex(md2, tables2):
                                        return md2, tables2, f"{key}-follow"

                                # PDF면 텍스트 추출
                                if is_pdf_by_header2 or is_pdf_by_magic2 or u2.lower().endswith(".pdf"):
                                    try:
                                        _load_annex_pdf_cache()
                                        content2 = resp2.content or b""
                                        h2 = hashlib.sha256(content2).hexdigest() if content2 else None
                                        cached2 = _ANNEX_PDF_CACHE.get(u2) if h2 else None
                                        if cached2 and cached2.get("hash") == h2 and cached2.get("text"):
                                            return cached2.get("text"), None, "pdf"
                                        text2 = extract_text(BytesIO(content2))
                                    except PDFSyntaxError:
                                        text2 = ""
                                    if (text2 or "").strip():
                                        out_text2 = _clean_md(text2)
                                        if h2:
                                            _ANNEX_PDF_CACHE[u2] = {"hash": h2, "text": out_text2}
                                            _ANNEX_PDF_CACHE_DIRTY = True
                                        return out_text2, None, "pdf"
                            except Exception:
                                # 후보 하나 실패 → 다음 후보 시도
                                continue

                        # 후보들 전부 실패 → 다음 링크(detail/html/pdf)로
                        continue
                    except Exception:
                        # 파싱 실패 → 다음 링크(detail/html/pdf)로
                        continue

                    # 후보들 전부 실패 → 다음 링크(detail/html/pdf)로 넘어감
                # 여기까지 왔으면 의미 있는 HTML
                return md, tables, key


            # PDF 계열 처리
            if key == "pdf" or is_pdf_by_header or is_pdf_by_magic or url.lower().endswith(".pdf"):
                _load_annex_pdf_cache()
                try:
                    content = resp.content or b""
                    h = hashlib.sha256(content).hexdigest() if content else None
                    cached = _ANNEX_PDF_CACHE.get(url) if h else None
                    if cached and cached.get("hash") == h and cached.get("text"):
                        return cached.get("text"), None, key
                    text = extract_text(BytesIO(content))
                except PDFSyntaxError:
                    text = ""
                if (text or "").strip():
                    out_text = _clean_md(text)
                    if h:
                        _ANNEX_PDF_CACHE[url] = {"hash": h, "text": out_text}
                        _ANNEX_PDF_CACHE_DIRTY = True
                    return out_text, None, key

        except Exception:
            # 현재 링크 처리 실패 → 다음 후보 링크로 넘어감
            continue

    # 어떤 링크에서도 본문을 만들지 못한 경우
    return None, None, None


try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # noqa

from utils import LawAPIClient

# ----------------------- 기본 설정 -----------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
PRETTY_NAMES = os.getenv("PRETTY_NAMES", "1") == "1"      # 한글+MST 파일명 사용
BUNDLE_PER_LAW = os.getenv("BUNDLE_PER_LAW", "0") == "1"  # 법령별 폴더 묶음 저장
STRIP_HEADINGS = os.getenv("STRIP_HEADINGS", "1") == "1"  # 편/장/절/관/부칙/별표/서식 제거

# ----------------------- 안전 파일 쓰기 -------------------
class AtomicWriter:
    def __init__(self, final_path: str):
        self.final_path = final_path
        self.tmp_fd = None
        self.tmp_path = None

    def __enter__(self):
        d = os.path.dirname(self.final_path)
        os.makedirs(d, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(self.final_path) + ".", suffix=".tmp", dir=d or None)
        self.tmp_fd = fd
        self.tmp_path = tmp_path
        return self

    def write(self, data: bytes):
        assert self.tmp_fd is not None
        os.write(self.tmp_fd, data)

    def __exit__(self, exc_type, exc, tb):
        if self.tmp_fd is not None:
            os.close(self.tmp_fd)
        if exc is None and self.tmp_path is not None:
            os.replace(self.tmp_path, self.final_path)
        else:
            if self.tmp_path and os.path.exists(self.tmp_path):
                try:
                    os.remove(self.tmp_path)
                except Exception:
                    pass

# ----------------------- 임베딩 (안전한 플레이스홀더) -------------
import hashlib

def embed_text(text: str) -> List[float]:
    """해시 → [−1, 1] 범위의 8차원 벡터로 매핑하고 L2 정규화"""
    h = hashlib.sha256((text or "").encode("utf-8")).digest()  # 32 bytes
    # 32바이트 → 8개 uint32
    vals = [int.from_bytes(h[i:i+4], "big", signed=False) for i in range(0, 32, 4)]
    a = np.array(vals, dtype="float32") / 4294967295.0  # [0,1]
    a = 2.0 * a - 1.0                                   # [-1,1]
    # L2 normalize (영벡터 보호)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        a[0] = 1e-6
        n = 1.0
    a = a / n
    return a.astype("float32").tolist()
def _sanitize_vec(v):
    import numpy as np
    a = np.asarray(v, dtype="float32")
    # NaN/Inf -> 0.0
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    # 영벡터 방지(완전 0이면 첫 요소에 아주 작은 값)
    if a.size and float(np.linalg.norm(a)) == 0.0:
        a[0] = 1e-6
    return a

# ----------------------- 헬퍼 -----------------------------
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _sg(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

def _clean(s):
    return " ".join(str(s).split()) if s else ""

def _fs_slug(name: str, maxlen: int = 80) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFC", str(name))
    s = re.sub(r'[\\/:*?"<>|]+', " ", s)
    s = "".join(ch for ch in s if ch.isprintable())
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > maxlen:
        s = s[:maxlen].rstrip()
    return s

def _get_law_korean_name(law_json: dict) -> Optional[str]:
    if not isinstance(law_json, dict):
        return None
    law = law_json.get("법령") if isinstance(law_json.get("법령"), dict) else law_json
    preferred = [
        "법령명한글",
        "법령명",
        "법령약칭",
    ]
    for k in preferred:
        v = law.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "법령명" in str(k):
                    if isinstance(v, (str, int)) and str(v).strip():
                        return str(v).strip()
                got = walk(v)
                if got:
                    return got
        elif isinstance(obj, list):
            for it in obj:
                got = walk(it)
                if got:
                    return got
        return None
    return walk(law)


def _clean(s):
    return " ".join(str(s).split()) if s else ""

def _find_korean_name_from_laws_dir(mst: str) -> Optional[str]:
    """laws/*.json의 목록 파일에서 MST에 해당하는 한글명을 폴백으로 찾는다."""
    try:
        laws_dir = "laws"
        if not os.path.isdir(laws_dir):
            return None
        for fn in os.listdir(laws_dir):
            if not fn.lower().endswith(".json"):
                continue
            p = os.path.join(laws_dir, fn)
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            law_items = (data.get("LawSearch", {}) or {}).get("law", [])
            if not isinstance(law_items, list):
                law_items = [law_items]
            for item in law_items:
                if str(item.get("법령일련번호")) == str(mst):
                    return item.get("법령약칭명") or item.get("법령명_한글") or item.get("법령명")
    except Exception:
        return None
    return None

def _make_base_name(mst: str, law_json: dict) -> str:
    try:
        nm = _get_law_korean_name(law_json)
        if not nm:
            nm = _find_korean_name_from_laws_dir(mst)
        return f"{_fs_slug(nm)}_{mst}" if nm else str(mst)
    except Exception:
        return str(mst)

def _ensure_meta(dst_law: dict, src_law: dict) -> None:
    """dst_law에 src_law의 주요 메타(법령명 등)를 비어있을 때만 채워 넣는다."""
    if not isinstance(dst_law, dict) or not isinstance(src_law, dict):
        return
    for k in ("법령일련번호", "법령약칭명", "법령명_한글", "법령명", "공포일자", "시행일자"):
        if dst_law.get(k) is None and src_law.get(k) is not None:
            dst_law[k] = src_law[k]

# ----------------------- 응답 정규화/평탄화 ----------------
def _normalize_law(data: Dict[str, Any]) -> Dict[str, Any]:
    """다양한 응답을 {"법령": { ... , "조문":[...] }}로 통일."""
    if not isinstance(data, dict):
        return {"법령": {"조문": []}}
    law = data.get("법령", data)
    if isinstance(law, list):
        law = law[0] if law else {}
    if not isinstance(law, dict):
        law = {}
    arts = law.get("조문")
    # 일부는 {"조문":{"조문":[...]}} 형태
    if isinstance(arts, dict) and "조문" in arts:
        arts = arts["조문"]
    law["조문"] = _as_list(arts)
    return {"법령": law}

def _get_articles_any_shape(law_json):
    """
    법령['조문']이 다음 중 무엇이든 모두 '실제 조문' 리스트로 평탄화:
    - [ {조문번호/조문내용/항...}, ... ] (전통형)
    - [ {"조문단위":[ {...}, {...} ]}, ... ] (래퍼형)
    - {"조문단위":[ ... ]} (dict 단일)
    """
    law = _sg(law_json, "법령") or law_json
    raw = _sg(law, "조문")
    items: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        if "조문단위" in raw:
            items.extend(_as_list(raw["조문단위"]))
        else:
            items.append(raw)
        return items
    for a in _as_list(raw):
        if isinstance(a, dict) and "조문단위" in a:
            items.extend(_as_list(a["조문단위"]))
        else:
            items.append(a)
    return items

_heading_re = re.compile(r"^(제\d+(편|장|절|관)\b|부칙\b|별표\b|서식\b)")

def _is_heading_only(art: dict) -> bool:
    """항/호/목이 없고 제목/본문이 '편/장/절/관/부칙/별표/서식'인 헤딩인지."""
    if not STRIP_HEADINGS:
        return False
    has_hang = bool(_sg(art, "항"))
    if has_hang:
        return False
    title = _clean(_sg(art, "조문제목"))
    body  = _clean(_sg(art, "조문내용"))
    s = title or body
    if not s:
        return True
    return bool(_heading_re.match(s))

# ----------------------- 구조화 추출(조/항/목) ------------
# ----------------------- 구조화 추출(조/항/목) ------------
def extract_units(law_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    본문(조/항/목) 유닛화.
    - 조(level='조'): 조문번호/제목/내용
    - 항(level='항'): 항번호/내용
    - 목(level='목'): 목(또는 호) 번호/내용
    path 예시: "제14조", "제14조 > 제③항", "제14조 > 제③항 > 2.호"
    """
    units: List[Dict[str, Any]] = []

    def _stringify(v) -> str:
        # dict/list/str 어떤 형태든 텍스트로 안전 변환
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            parts = []
            for vv in v.values():
                s = _stringify(vv).strip()
                if s:
                    parts.append(s)
            return "\n".join(parts)
        if isinstance(v, (list, tuple)):
            parts = []
            for vv in v:
                s = _stringify(vv).strip()
                if s:
                    parts.append(s)
            return "\n".join(parts)
        return str(v or "")

    def _first_value(obj, keys: List[str]) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        for k in keys:
            v = _sg(obj, k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return None

    def _list_from(obj: Any, key: str) -> List[Any]:
        raw = _sg(obj, key)
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
        return _as_list(raw)

    def _item_path_join(*parts: Optional[str]) -> Optional[str]:
        cleaned = [p for p in parts if p]
        return "-".join(cleaned) if cleaned else None

    articles = _get_articles_any_shape(law_json)  # 다양한 형태의 '조문'을 평탄화
    for art in articles:
        if _is_heading_only(art):  # 편/장/절/관/부칙/별표/서식 등 헤딩만인 경우 스킵
            continue

        jo_raw   = _clean(_sg(art, "조문번호"))
        jo_title = _clean(_sg(art, "조문제목"))
        jo_body  = _clean(_stringify(_sg(art, "조문내용")))
        jo_id = _first_value(art, ["조문일련번호"])

        # 1) 조 단위
        if jo_raw and (jo_title or jo_body):
            units.append({
                "level": "조",
                "jo": jo_raw, "hang": None, "mok": None,
                "title": jo_title or None,
                "text": jo_body or "",
                "path": f"제{jo_raw}조",
                "article_id": jo_id,
                "paragraph_id": None,
                "item_id": None,
            })

        # 2) 항 단위
        for h in _list_from(art, "항"):
            hang_no   = _clean(_sg(h, "항번호") or _sg(h, "항"))
            hang_body = _clean(_stringify(_sg(h, "항내용") or h))
            if not hang_no or not hang_body:
                continue
            hang_id = _first_value(h, ["항일련번호"])
            h_path = f"제{jo_raw}조 > 제{hang_no}항" if jo_raw else f"제{hang_no}항"
            units.append({
                "level": "항",
                "jo": jo_raw, "hang": hang_no, "mok": None,
                "title": None,
                "text": hang_body,
                "path": h_path,
                "article_id": jo_id,
                "paragraph_id": hang_id,
                "item_id": None,
            })

            # 3) 호 단위
            for ho in _list_from(h, "호"):
                ho_no   = _clean(_sg(ho, "호번호") or _sg(ho, "호"))
                ho_body = _clean(_stringify(_sg(ho, "호내용") or ho))
                if not ho_no or not ho_body:
                    continue
                ho_id = _first_value(ho, ["호일련번호"])
                ho_path_raw = _item_path_join(ho_no)
                units.append({
                    "level": "목",
                    "jo": jo_raw, "hang": hang_no, "mok": ho_no,
                    "title": None,
                    "text": ho_body,
                    "path": f"{h_path} > {ho_no}",
                    "item_type": "호",
                    "item_path_raw": ho_path_raw,
                    "article_id": jo_id,
                    "paragraph_id": hang_id,
                    "item_id": ho_id,
                })

                # 3-1) 호 > 목 단위
                for mok in _list_from(ho, "목"):
                    mok_no   = _clean(_sg(mok, "목번호") or _sg(mok, "목"))
                    mok_body = _clean(_stringify(_sg(mok, "목내용") or mok))
                    if not mok_no or not mok_body:
                        continue
                    mok_id = _first_value(mok, ["목일련번호"])
                    mok_path_raw = _item_path_join(ho_no, mok_no)
                    units.append({
                        "level": "목",
                        "jo": jo_raw, "hang": hang_no, "mok": mok_no,
                        "title": None,
                        "text": mok_body,
                        "path": f"{h_path} > {ho_no} > {mok_no}",
                        "item_type": "목",
                        "item_path_raw": mok_path_raw,
                        "article_id": jo_id,
                        "paragraph_id": hang_id,
                        "item_id": mok_id,
                    })

                    # 3-2) 호 > 목 > 세목 단위
                    for semok in _list_from(mok, "세목"):
                        sm_no   = _clean(_sg(semok, "세목번호") or _sg(semok, "세목"))
                        sm_body = _clean(_stringify(_sg(semok, "세목내용") or semok))
                        if not sm_no or not sm_body:
                            continue
                        sm_id = _first_value(semok, ["세목일련번호"])
                        sm_path_raw = _item_path_join(ho_no, mok_no, sm_no)
                        units.append({
                            "level": "목",
                            "jo": jo_raw, "hang": hang_no, "mok": sm_no,
                            "title": None,
                            "text": sm_body,
                            "path": f"{h_path} > {ho_no} > {mok_no} > {sm_no}",
                            "item_type": "세목",
                            "item_path_raw": sm_path_raw,
                            "article_id": jo_id,
                            "paragraph_id": hang_id,
                            "item_id": sm_id,
                        })

            # 4) 항 > 목 단위 (호 없이 직접 목이 오는 경우)
            for mok in _list_from(h, "목"):
                mok_no   = _clean(_sg(mok, "목번호") or _sg(mok, "목"))
                mok_body = _clean(_stringify(_sg(mok, "목내용") or mok))
                if not mok_no or not mok_body:
                    continue
                mok_id = _first_value(mok, ["목일련번호"])
                mok_path_raw = _item_path_join(mok_no)
                units.append({
                    "level": "목",
                    "jo": jo_raw, "hang": hang_no, "mok": mok_no,
                    "title": None,
                    "text": mok_body,
                    "path": f"{h_path} > {mok_no}",
                    "item_type": "목",
                    "item_path_raw": mok_path_raw,
                    "article_id": jo_id,
                    "paragraph_id": hang_id,
                    "item_id": mok_id,
                })

                for semok in _list_from(mok, "세목"):
                    sm_no   = _clean(_sg(semok, "세목번호") or _sg(semok, "세목"))
                    sm_body = _clean(_stringify(_sg(semok, "세목내용") or semok))
                    if not sm_no or not sm_body:
                        continue
                    sm_id = _first_value(semok, ["세목일련번호"])
                    sm_path_raw = _item_path_join(mok_no, sm_no)
                    units.append({
                        "level": "목",
                        "jo": jo_raw, "hang": hang_no, "mok": sm_no,
                        "title": None,
                        "text": sm_body,
                        "path": f"{h_path} > {mok_no} > {sm_no}",
                        "item_type": "세목",
                        "item_path_raw": sm_path_raw,
                        "article_id": jo_id,
                        "paragraph_id": hang_id,
                        "item_id": sm_id,
                    })

    return units

def extract_buchik_units(law_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    현행 본문 JSON의 '부칙' 블록을 간단히 유닛화.
    level='부칙', path='부칙', title/text 채워서 반환.
    """
    units: List[Dict[str, Any]] = []
    try:
        law = law_json.get("법령") or law_json
        buk = law.get("부칙")
        if not isinstance(buk, dict):
            return units

        items = buk.get("부칙단위") or []
        if isinstance(items, dict):
            items = [items]

        for it in items:
            if not isinstance(it, dict):
                continue
            title = (it.get("부칙제목") or "부칙").strip()

            contents: List[str] = []
            content_blocks = it.get("부칙내용") or []
            if isinstance(content_blocks, dict):
                content_blocks = [content_blocks]
            if not isinstance(content_blocks, list):
                content_blocks = [content_blocks]

            for blk in content_blocks:
                if isinstance(blk, dict):
                    iter_lines = list(blk.values())
                elif isinstance(blk, list):
                    iter_lines = blk
                else:
                    iter_lines = [blk]
                for ln in iter_lines:
                    s = str(ln).strip()
                    if s:
                        contents.append(s)

            text = "\n".join(contents).strip()
            if not text:
                continue

            units.append({
                "level": "부칙",
                "jo": None, "hang": None, "mok": None,
                "title": title,
                "text": text,
                "path": "부칙",
                "article_id": None,
                "paragraph_id": None,
                "item_id": None,
            })
    except Exception:
        return units
    return units


def _norm_title(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("·", "").replace(".", "").replace("(", "").replace(")", "")
    return s

def _loose_match_title(a: str, b: str) -> bool:
    if not a or not b:
        return False
    A = _norm_title(a)
    B = _norm_title(b)
    return A in B or B in A

def fetch_annex_units(client: LawAPIClient, mst: str, law_title: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    licbyl(별표/서식) 검색 → 해당 MST/법령명으로 부속(별표/서식)만 얇게 수집.
    최소 버전: 제목/번호/링크만 채움 (본문 텍스트는 제목 복제)
    """
    units: List[Dict[str, Any]] = []
    expected = {"annex_no": set(), "annex_id": set()}

    # 1) licbyl 검색 (해당 법령명 기반, 느슨 매칭 허용)
    rows: List[Dict[str, Any]] = []
    try:
        qlaw = law_title
        if not qlaw:
            try:
                law_json = client.get_law(mst)
                qlaw = _get_law_korean_name(law_json) or ""
            except Exception:
                qlaw = ""
        if qlaw:
            page = 1
            # ------ [PATCH: licbyl paging guards] ------
            MAX_PAGES = 50              # 안전 상한 (원하면 30~100 사이로 조절)
            NOHIT_STOP_AFTER = 3        # 연속 N페이지 매칭 0건이면 조기 종료
            KEEP_LIMIT = 80             # 충분히 많이 모였으면 조기 종료
            keep_incremental: List[Dict[str, Any]] = []
            nohit_streak = 0
            # ------ [PATCH END] ------
            while True:
                resp = client.search_attachments(query=qlaw, page=page, display=100, sort="lasc", search=2)

                # 응답 어디에 있든 licbyl 리스트만 뽑기
                def _rows(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
                    if not isinstance(resp, dict):
                        return []
                    blk = resp.get("licBylSearch") or resp.get("LicBylSearch") or resp.get("licbylsearch")
                    rows0 = (blk.get("licbyl") if isinstance(blk, dict) else None) or resp.get("licbyl")
                    if rows0:
                        return rows0 if isinstance(rows0, list) else [rows0]
                    out: List[Dict[str, Any]] = []
                    stack = [resp]
                    while stack:
                        cur = stack.pop()
                        if isinstance(cur, dict):
                            for k, v in cur.items():
                                if k.lower() == "licbyl":
                                    if isinstance(v, list):
                                        out.extend(v)
                                    elif isinstance(v, dict):
                                        out.append(v)
                                elif isinstance(v, (dict, list)):
                                    stack.append(v)
                        elif isinstance(cur, list):
                            stack.extend(cur)
                    return out

                cur_rows = _rows(resp)
                # ------ [PATCH: incremental filter & early stop] ------
                new_keep: List[Dict[str, Any]] = []
                for r in cur_rows:
                    law_nm = str(r.get("관련법령명") or r.get("법령명") or "").strip()
                    rel = str(r.get("관련법령일련번호") or r.get("법령일련번호") or r.get("MST") or "").strip()
                    if (mst and rel and rel == str(mst)) or (law_title and law_nm and _loose_match_title(law_title, law_nm)):
                        new_keep.append(r)

                if new_keep:
                    keep_incremental.extend(new_keep)
                    nohit_streak = 0
                else:
                    nohit_streak += 1
                # ------ [PATCH END] ------
                rows.extend(cur_rows)

                blk = resp.get("licBylSearch") or {}
                total = int(str(blk.get("totalCnt") or len(cur_rows)))
                per = int(str(blk.get("numOfRows") or 100))
                if page * per >= total:
                    break
                # ------ [PATCH: hard cap & early stop] ------
                if page >= MAX_PAGES:
                    logger.info("[annex] page cap reached at %d (totalCnt=%s)", page, blk.get("totalCnt"))
                    break
                if len(keep_incremental) >= KEEP_LIMIT or nohit_streak >= NOHIT_STOP_AFTER:
                    logger.info(
                        "[annex] early stop paging: keep=%d nohit=%d page=%d",
                        len(keep_incremental), nohit_streak, page
                    )
                    break
                # ------ [PATCH END] ------
                page += 1
    except Exception:
        logger.exception("[annex] licbyl search failed")
        rows = []
        # ------ [PATCH: use filtered rows if any] ------
        if keep_incremental:
            rows = keep_incremental
            # ------ [PATCH END] ------


    # licbyl 행 필터링: MST 일치 OR 법령명 느슨 매칭
    keep: List[Dict[str, Any]] = []
    for r in rows:
        law_nm = str(r.get("관련법령명") or r.get("법령명") or "").strip()
        rel = str(r.get("관련법령일련번호") or r.get("법령일련번호") or r.get("MST") or "").strip()
        if mst and rel and rel == str(mst):
            keep.append(r); continue
        if qlaw and law_nm and _loose_match_title(qlaw, law_nm):
            keep.append(r); continue
    rows = keep
    logger.info("[annex] rows after loose match = %d", len(rows))


    # licbyl 행 → 유닛화
    for r in rows:
        kind = str(r.get("별표종류") or "별표")
        no = str(r.get("별표번호") or "").strip()
        title = _clean(str(r.get("별표명") or ""))
        human = None
        if no and re.fullmatch(r"\d{6}", no):
            a = int(no[:4]); b = int(no[4:])
            human = f"{kind} {a}" + (f"의{b}" if b else "")
        links = {
            "detail": r.get("별표법령상세링크"),
            "html": r.get("별표서식파일링크"),
            "pdf": r.get("별표서식PDF파일링크"),
        }
        annex_id = str(r.get("별표일련번호") or "").strip() or None
        units.append({
            "level": kind,
            "annex_no": no or None,
            "annex_no_human": human,
            "title": title,
            "text": title,  # 최소 버전
            "links": links,
            "_annex_meta": {"extracted_from": None},
            "path": f"{kind}",
            "annex_id": annex_id,
            "article_id": None,
            "paragraph_id": None,
            "item_id": None,
        })

    # 2) 폴백: licbyl이 비었으면 조문제목에서 [별표]/[서식] 감지
    if not units:
        try:
            law_json = client.get_law(mst)
            def _parse_from_title(t: str):
                m = re.search(r"(별표|서식)\s*([0-9]+)(?:\s*의\s*([0-9]+))?", t)
                if not m:
                    return None
                kind = m.group(1); a = int(m.group(2)); b = int(m.group(3) or 0)
                code = f"{a:04d}{b:02d}"
                human = f"{kind} {a}" + (f"의{b}" if b else "")
                return kind, human, code
            articles = _get_articles_any_shape(law_json)
            for art in articles:
                t = _clean(str((art or {}).get("조문제목") or ""))
                if not t:
                    continue
                parsed = _parse_from_title(t)
                if not parsed:
                    continue
                kind, human, code = parsed
                units.append({
                    "level": kind,
                    "annex_no": code,
                    "annex_no_human": human,
                    "title": t,
                    "text": t,
                    "links": {},
                    "_annex_meta": {"extracted_from": "title"},
                    "path": f"{kind}",
                    "annex_id": None,
                    "article_id": None,
                    "paragraph_id": None,
                    "item_id": None,
                })
        except Exception:
            logger.exception("[annex] fallback from titles failed")

    # 3) dedup: annex_no 기준, 링크가 많은 쪽 우선
    by: Dict[str, Dict[str, Any]] = {}
    def _score(u: Dict[str, Any]) -> int:
        ln = (u.get("links") or {})
        return (1 if ln.get("detail") else 0) + (2 if ln.get("html") else 0) + (3 if ln.get("pdf") else 0)
    for u in units:
        key = u.get("annex_no") or (u.get("level"), u.get("title"))
        key = str(key)
        if key not in by or _score(u) > _score(by[key]):
            by[key] = u
    
    # 4) 본문 주입: licbyl-JSON → detail/html → pdf 순으로 시도
    out = list(by.values())
    for u in out:
        if (u.get("level") not in ("별표", "서식")):
            continue
        links = u.get("links") or {}
        try:
            md, tables, src = _fetch_annex_body_with_links(client, links)
            if md:
                u["text"] = md
                meta = u.get("_annex_meta") or {}
                if src:
                    meta["source"] = src
                if tables:
                    meta["tables"] = tables
                u["_annex_meta"] = meta
                logger.info("[annex] filled text for %s via %s", u.get("annex_no") or u.get("title"), src)

        except Exception:
            # 본문 추출 실패 시 최소버전(제목 그대로) 유지
            pass

    if not expected["annex_no"]:
        for u in out:
            if u.get("annex_no"):
                expected["annex_no"].add(str(u.get("annex_no")))
    if not expected["annex_id"]:
        for u in out:
            if u.get("annex_id"):
                expected["annex_id"].add(str(u.get("annex_id")))

    _save_annex_pdf_cache()
    return out, expected


# ----------------------- 수집(메타 보존+페이징) -----------
def fetch_full_law(client: LawAPIClient, mst: str) -> Dict[str, Any]:
    """
    1) 전체 호출 → 조문/본문이 있으면 그대로
    2) 부족하면 목록→상세(JO 후보)로 병합
    3) 그래도 부족하면 JO=000100,000200,... 페이징 (연속 빈 응답 n회면 중단)
       - 상한: LAW_JO_MAX_PAGES(기본 80), 빈연속: LAW_JO_EMPTY_STREAK(기본 5)
    4) 최종 {"법령": {...}} 반환 + 디버그 JSON 저장
    """
    max_pages = int(os.getenv("LAW_JO_MAX_PAGES", "80"))
    empty_streak_max = int(os.getenv("LAW_JO_EMPTY_STREAK", "5"))

    # 1) 기본
    base = client.get_law(mst)
    try:
        if list(extract_units(_normalize_law(base))):
            merged0 = _normalize_law(base)
            # base 메타 보존
            _ensure_meta(merged0["법령"], _normalize_law(base)["법령"])
            _dump_debug_json(f"laws/debug_law_{mst}.json", merged0)
            return merged0
    except Exception:
        pass

    # 2) 목록→상세 (가능하면)
    merged = _normalize_law(base)
    _ensure_meta(merged["법령"], _normalize_law(base)["법령"])  # ★ base 메타 주입

    jos = _iter_jo_numbers_for_list_detail(merged)  # 목록 기반 JO 후보
    if jos:
        for jo in jos:
            try:
                part = client.get_law(mst, jo=jo)
                dst = merged["법령"]
                src = _normalize_law(part)["법령"]
                _ensure_meta(dst, src)  # ★ 상세의 메타도 채우기
                dst_arts = _as_list(dst.get("조문"))
                src_arts = _as_list(src.get("조문"))
                before = len(dst_arts)
                dst_arts.extend(src_arts)
                dst["조문"] = dst_arts
                added = len(dst_arts) - before
                if added:
                    logger.info(f"MST {mst} JO={jo} merged {added} (total={len(dst_arts)})")
            except Exception as e:
                logger.warning(f"MST {mst} JO={jo} merge failed (list→detail): {e} (continue)")
        try:
            if list(extract_units(merged)):
                _dump_debug_json(f"laws/debug_law_{mst}.json", merged)
                return merged
        except Exception:
            pass

    # 3) JO 페이징 폴백
    empty_streak = 0
    for i in range(1, max_pages + 1):
        jo6 = f"{i:04d}00"  # 000100, 000200, ...
        try:
            part = client.get_law(mst, jo=jo6)
            src = _normalize_law(part)["법령"]
            _ensure_meta(merged["법령"], src)  # ★ JO의 메타도 채우기
            src_arts = _as_list(src.get("조문"))
            if not src_arts:
                empty_streak += 1
                logger.info(f"MST {mst} JO={jo6} merged 0 (empty={empty_streak})")
                if empty_streak >= empty_streak_max:
                    logger.info(f"MST {mst} stop paging after {empty_streak} consecutive empties")
                    break
                continue
            empty_streak = 0
            dst = merged["법령"]
            dst_arts = _as_list(dst.get("조문"))
            before = len(dst_arts)
            dst_arts.extend(src_arts)
            dst["조문"] = dst_arts
            added = len(dst_arts) - before
            logger.info(f"MST {mst} JO={jo6} merged {added} (total={len(dst_arts)})")
        except Exception as e:
            logger.warning(f"MST {mst} JO={jo6} fetch failed: {e} (continue)")

    # laws/*.json 폴더 폴백으로 이름 주입(없을 때만)
    nm = _find_korean_name_from_laws_dir(mst)
    if nm and not (_sg(merged["법령"], "법령약칭명") or _sg(merged["법령"], "법령명_한글") or _sg(merged["법령"], "법령명")):
        merged["법령"]["법령명_한글"] = nm

    # 결과 검증/저장
    if not list(extract_units(merged)):
        raise RuntimeError(f"No parsable articles for MST {mst}")
    _dump_debug_json(f"laws/debug_law_{mst}.json", merged)
    return merged

def _iter_jo_numbers_for_list_detail(law_json: Dict[str, Any]) -> List[str]:
    """목록의 조문번호에서 JO 후보(6자리) 추출. 목록→상세 병합에 사용."""
    jos: List[str] = []
    seen = set()
    for art in _get_articles_any_shape(law_json):
        raw = str(_sg(art, "조문번호") or _sg(art, "조문일련번호") or "").strip()
        if not raw:
            continue
        # '10' -> 001000, '10의2' -> 001002
        m = re.match(r"^\s*(\d+)(?:\s*의\s*(\d+))?\s*$", raw)
        if m:
            main = int(m.group(1))
            sub = int(m.group(2) or 0)
            jo = f"{main:04d}{sub:02d}"
        else:
            digits = re.findall(r"\d+", raw)
            if not digits:
                continue
            main = int(digits[0]); sub = int(digits[1]) if len(digits) > 1 else 0
            jo = f"{main:04d}{sub:02d}"
        if jo not in seen:
            seen.add(jo); jos.append(jo)
    return jos

# ----------------------- 디버그 저장 ----------------------
def _dump_debug_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ----------------------- 무결성 검증 ----------------------
def _list_from(obj: Any, key: str) -> List[Any]:
    raw = _sg(obj, key)
    if isinstance(raw, dict) and key in raw:
        raw = raw[key]
    return _as_list(raw)

def _collect_source_id_sets(law_json: Dict[str, Any]) -> Dict[str, set]:
    ids = {"article": set(), "paragraph": set(), "item": set()}
    for art in _get_articles_any_shape(law_json):
        if not isinstance(art, dict):
            continue
        jo_id = _sg(art, "조문일련번호")
        if jo_id:
            ids["article"].add(str(jo_id))
        for h in _list_from(art, "항"):
            if not isinstance(h, dict):
                continue
            hang_id = _sg(h, "항일련번호")
            if hang_id:
                ids["paragraph"].add(str(hang_id))
            for ho in _list_from(h, "호"):
                if not isinstance(ho, dict):
                    continue
                ho_id = _sg(ho, "호일련번호")
                if ho_id:
                    ids["item"].add(str(ho_id))
                for mok in _list_from(ho, "목"):
                    if not isinstance(mok, dict):
                        continue
                    mok_id = _sg(mok, "목일련번호")
                    if mok_id:
                        ids["item"].add(str(mok_id))
                    for semok in _list_from(mok, "세목"):
                        if not isinstance(semok, dict):
                            continue
                        sm_id = _sg(semok, "세목일련번호")
                        if sm_id:
                            ids["item"].add(str(sm_id))
            for mok in _list_from(h, "목"):
                if not isinstance(mok, dict):
                    continue
                mok_id = _sg(mok, "목일련번호")
                if mok_id:
                    ids["item"].add(str(mok_id))
                for semok in _list_from(mok, "세목"):
                    if not isinstance(semok, dict):
                        continue
                    sm_id = _sg(semok, "세목일련번호")
                    if sm_id:
                        ids["item"].add(str(sm_id))
    return ids

def _find_dups(values: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}

def _annex_base_missing(annex_no_set: set) -> List[str]:
    base_to_suffix: Dict[int, set] = {}
    for code in annex_no_set:
        s = str(code).strip()
        if not s.isdigit() or len(s) != 6:
            continue
        base = int(s[:4]); suf = int(s[4:])
        base_to_suffix.setdefault(base, set()).add(suf)
    missing = []
    for base, sufs in base_to_suffix.items():
        if sufs and 0 not in sufs:
            missing.append(f"{base:04d}00")
    return missing

def _make_failure_entry(
    failure_type: str,
    mst: Optional[str] = None,
    crawl_ts: Optional[str] = None,
    seed: Optional[int] = None,
    unit: Optional[Dict[str, Any]] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "failure_type": failure_type,
        "mst": mst,
        "crawl_ts": crawl_ts,
        "seed": seed,
        "source_anchor": None,
        "display_path_norm": None,
        "source_url": None,
        "level": None,
        "ref": None,
    }
    if unit:
        entry["source_anchor"] = unit.get("source_anchor")
        entry["display_path_norm"] = unit.get("display_path_norm")
        entry["source_url"] = unit.get("source_url")
        entry["level"] = unit.get("level")
        entry["ref"] = {
            "annex_refs": unit.get("annex_refs"),
            "ref_by": unit.get("ref_by"),
        }
    if detail:
        entry["detail"] = detail
    return entry


def _validate_units_integrity(
    law_json: Dict[str, Any],
    units: List[Dict[str, Any]],
    annex_expected: Optional[Dict[str, Any]] = None,
    mst: Optional[str] = None,
    crawl_ts: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    source_ids = _collect_source_id_sets(law_json)

    unit_article_ids = [str(u.get("article_id")) for u in units if u.get("article_id")]
    unit_paragraph_ids = [str(u.get("paragraph_id")) for u in units if u.get("paragraph_id")]
    unit_item_ids = [str(u.get("item_id")) for u in units if u.get("item_id")]
    unit_annex_ids = [str(u.get("annex_id")) for u in units if u.get("annex_id") and not u.get("annex_placeholder")]
    unit_annex_nos = [str(u.get("annex_no")) for u in units if u.get("annex_no") and not u.get("annex_placeholder")]

    unit_sets = {
        "article": set(unit_article_ids),
        "paragraph": set(unit_paragraph_ids),
        "item": set(unit_item_ids),
        "annex_id": set(unit_annex_ids),
        "annex_no": set(unit_annex_nos),
    }

    id_check_active = any(source_ids.get(k) for k in ("article", "paragraph", "item"))
    missing_ids = {}
    extra_ids = {}
    for k in ("article", "paragraph", "item"):
        if id_check_active and source_ids.get(k):
            missing_ids[k] = sorted(source_ids[k] - unit_sets[k])
            extra_ids[k] = sorted(unit_sets[k] - source_ids[k])
        else:
            missing_ids[k] = []
            extra_ids[k] = []

    annex_expected = annex_expected or {}
    expected_annex_no = set(annex_expected.get("annex_no") or [])
    expected_annex_id = set(annex_expected.get("annex_id") or [])
    missing_annex_no = sorted(expected_annex_no - unit_sets["annex_no"]) if expected_annex_no else []
    missing_annex_id = sorted(expected_annex_id - unit_sets["annex_id"]) if expected_annex_id else []
    extra_annex_no = sorted(unit_sets["annex_no"] - expected_annex_no) if expected_annex_no else []
    extra_annex_id = sorted(unit_sets["annex_id"] - expected_annex_id) if expected_annex_id else []

    dups = {
        "article_id": _find_dups(unit_article_ids),
        "paragraph_id": _find_dups(unit_paragraph_ids),
        "item_id": _find_dups(unit_item_ids),
        "annex_id": _find_dups(unit_annex_ids),
        "annex_no": _find_dups(unit_annex_nos),
    }

    # 필수 메타 누락 점검
    missing_meta: List[Dict[str, Any]] = []
    for i, u in enumerate(units):
        missing = []
        if not u.get("law"):
            missing.append("law")
        if not u.get("mst"):
            missing.append("mst")
        if not u.get("source_anchor"):
            missing.append("source_anchor")
        if not u.get("source_url"):
            missing.append("source_url")
        lvl = u.get("level")
        if lvl in ("조", "항", "목") and not u.get("article"):
            missing.append("article")
        if lvl in ("항", "목") and not u.get("paragraph"):
            missing.append("paragraph")
        if lvl == "목" and not u.get("item_path"):
            missing.append("item_path")
        if missing:
            missing_meta.append({
                "idx": i,
                "level": lvl,
                "path": u.get("path"),
                "source_anchor": u.get("source_anchor"),
                "missing": missing,
            })

    annex_refs_missing: List[Dict[str, Any]] = []
    for u in units:
        if u.get("annex_refs_missing"):
            annex_refs_missing.append({
                "source_anchor": u.get("source_anchor"),
                "display_path_norm": u.get("display_path_norm"),
                "missing": u.get("annex_refs_missing"),
            })

    failures: List[Dict[str, Any]] = []
    unit_by_anchor = {u.get("source_anchor"): u for u in units if u.get("source_anchor")}
    for m in missing_meta:
        u = unit_by_anchor.get(m.get("source_anchor"))
        failures.append(_make_failure_entry(
            "missing_required_meta",
            mst=mst,
            crawl_ts=crawl_ts,
            seed=seed,
            unit=u,
            detail=m,
        ))
    for m in annex_refs_missing:
        u = unit_by_anchor.get(m.get("source_anchor"))
        failures.append(_make_failure_entry(
            "annex_ref_missing",
            mst=mst,
            crawl_ts=crawl_ts,
            seed=seed,
            unit=u,
            detail=m,
        ))
    if id_check_active:
        for k in ("article", "paragraph", "item"):
            if missing_ids.get(k):
                failures.append(_make_failure_entry(
                    "id_missing",
                    mst=mst,
                    crawl_ts=crawl_ts,
                    seed=seed,
                    unit=None,
                    detail={"id_type": k, "missing": missing_ids.get(k)},
                ))
            if extra_ids.get(k):
                failures.append(_make_failure_entry(
                    "id_extra",
                    mst=mst,
                    crawl_ts=crawl_ts,
                    seed=seed,
                    unit=None,
                    detail={"id_type": k, "extra": extra_ids.get(k)},
                ))
        for k, v in dups.items():
            if v:
                failures.append(_make_failure_entry(
                    "id_duplicate",
                    mst=mst,
                    crawl_ts=crawl_ts,
                    seed=seed,
                    unit=None,
                    detail={"id_type": k, "duplicate": v},
                ))
    report = {
        "source_counts": {k: len(v) for k, v in source_ids.items()},
        "unit_counts": {
            "article": len(unit_sets["article"]),
            "paragraph": len(unit_sets["paragraph"]),
            "item": len(unit_sets["item"]),
            "annex_id": len(unit_sets["annex_id"]),
            "annex_no": len(unit_sets["annex_no"]),
        },
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "duplicate_ids": dups,
        "annex_missing_no": missing_annex_no,
        "annex_missing_id": missing_annex_id,
        "annex_extra_no": extra_annex_no,
        "annex_extra_id": extra_annex_id,
        "annex_base_missing": _annex_base_missing(unit_sets["annex_no"]),
        "annex_refs_missing": annex_refs_missing,
        "missing_required_meta": missing_meta,
        "id_check": {
            "status": "active" if id_check_active else "skipped",
            "reason": None if id_check_active else "source_ids_missing",
        },
        "failures": failures,
    }
    return report

# ----------------------- 인덱스 빌드 ----------------------
def build_index_for_mst(mst: str, out_dir: str = "faiss_indexes", client: Optional[LawAPIClient] = None) -> Dict[str, Any]:
    client = client or LawAPIClient()
    t0 = time.monotonic()

    # 0) 수집
    law_json = fetch_full_law(client, mst)
    try:
        law_hash = hashlib.sha256(
            json.dumps(law_json, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    except Exception:
        law_hash = None

    # 1) 저장 경로(예쁜 파일명 적용)
    base = _make_base_name(mst, law_json) if PRETTY_NAMES else str(mst)
    os.makedirs(out_dir, exist_ok=True)

    if BUNDLE_PER_LAW:
        law_dir = os.path.join(out_dir, base)
        os.makedirs(law_dir, exist_ok=True)
        units_path   = os.path.join(law_dir, "units.json")
        answers_path = os.path.join(law_dir, "answers.json")
        idmap_path   = os.path.join(law_dir, "faiss_id_map.json")
        # ✅ 인덱스 파일만 ASCII(MST)로 out_dir에 저장
        index_path   = os.path.join(out_dir, f"{mst}_faiss_index.bin")
    else:
        units_path   = os.path.join(out_dir, f"{base}_units.json")
        answers_path = os.path.join(out_dir, f"{base}_answers.json")
        idmap_path   = os.path.join(out_dir, f"{base}_faiss_id_map.json")
        # ✅ 인덱스 파일만 ASCII(MST)로 저장
        index_path   = os.path.join(out_dir, f"{mst}_faiss_index.bin")

    # 2) 구조화 추출
    units = extract_units(law_json)

    # 2.1) 부칙 합류
    buchik = extract_buchik_units(law_json)
    if buchik:
        units.extend(buchik)
        logger.info(f"MST {mst}: 부칙 units += {len(buchik)}")

    # 2.2) 별표/서식 합류
    law_title = _get_law_korean_name(law_json)
    
    # 폴백: base가 "이름_MST" 형태면 앞부분을 법령명으로 사용
    if not law_title and isinstance(base, str):
        if base.endswith(f"_{mst}"):
            law_title = base[:-(len(mst) + 1)] or None

    annexes, annex_expected = fetch_annex_units(client, mst, law_title)
    if annexes:
        units.extend(annexes)
        logger.info(f"MST {mst}: 별표/서식 units += {len(annexes)}") 

    # 2.3) ✅ 후처리(한 번에)
    crawl_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    units = postprocess_units(
        units,
        law_meta=law_json.get("법령"),
        mst=str(mst),
        crawl_ts=crawl_ts,
        law_title=law_title,
    )
    if law_hash:
        for u in units:
            if not u.get("law_hash"):
                u["law_hash"] = law_hash

    # 2.4) 무결성 검증 (fail-fast)
    report = _validate_units_integrity(
        law_json,
        units,
        annex_expected,
        mst=str(mst),
        crawl_ts=crawl_ts,
        seed=None,
    )
    report["mst"] = str(mst)
    report["base"] = base
    report["crawl_ts"] = crawl_ts
    report["seed"] = None
    report["generated_at"] = crawl_ts
    has_errors = bool(report.get("failures"))
    if has_errors:
        if BUNDLE_PER_LAW:
            report_path = os.path.join(os.path.dirname(units_path), "validation_report.json")
        else:
            report_path = os.path.join(out_dir, f"{base}_validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        raise RuntimeError(f"Integrity validation failed for MST {mst}. Report: {report_path}")

    # 3) units 저장
    with AtomicWriter(units_path) as aw:
        aw.write(json.dumps(units, ensure_ascii=False, indent=2).encode('utf-8'))

    # 4) 임베딩 입력 & id_map
    texts_to_embed: List[str] = []
    id_map: List[Dict[str, Any]] = []
    for i, u in enumerate(units):
        norm_text = u.get("normalized_text") or u.get("text") or ""
        title = u.get("title") or ""
        combined = (title + "\n" if title else "") + norm_text
        combined = combined.strip()
        texts_to_embed.append(combined)
        id_map.append({
            "faiss_id": i,
            "mst": mst,
            "base": base,
            "level": u["level"],
            "jo": u.get("jo"),
            "hang": u.get("hang"),
            "mok": u.get("mok"),
            "path": u.get("path"),
            "title": u.get("title"),
        })

    # 5) answers 저장
    answers_payload = [
        {
            "id": i,
            "level": u["level"],
            "jo": u.get("jo"),
            "hang": u.get("hang"),
            "mok": u.get("mok"),
            "path": u.get("path"),
            "title": u.get("title"),
            "text": texts_to_embed[i],
        }
        for i, u in enumerate(units)
    ]
    with AtomicWriter(answers_path) as aw:
        aw.write(json.dumps(answers_payload, ensure_ascii=False, indent=2).encode('utf-8'))

    # 6) id_map 저장
    with AtomicWriter(idmap_path) as aw:
        aw.write(json.dumps(id_map, ensure_ascii=False, indent=2).encode('utf-8'))

    # 7) FAISS 인덱스 저장 (미설치 시 placeholder)
    if faiss is not None and texts_to_embed:
        # ✅ 각 벡터를 살균 후 스택
        xb = np.vstack([_sanitize_vec(embed_text(t)) for t in texts_to_embed]).astype('float32')
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)  # ✅ 안전모드: FLAT L2
        index.add(xb)
        faiss.write_index(index, index_path)
    else:
        with AtomicWriter(index_path) as aw:
            aw.write(b"FAISS_NOT_AVAILABLE")


    dt = time.monotonic() - t0
    logger.info("MST %s build finished in %.2fs (units=%d)", mst, dt, len(units))
       # 직전 본문에서 law_title을 이미 계산함: _get_law_korean_name(law_json)
    law_title = _get_law_korean_name(law_json)
    return {
        "mst": mst,
        "units": len(units),
        "duration_sec": dt,
        "out_dir": out_dir,
        "base": base,
        "law_title": law_title
    }
def build_indexes(msts: List[str], out_dir: str = "faiss_indexes", fail_fast: bool = True) -> List[Dict[str, Any]]:
    results = []
    oc = os.environ.get("LAW_API_OC")
    shared_client = LawAPIClient()
    for mst in msts:
        try:
            results.append(build_index_for_mst(mst, out_dir, client=shared_client))
        except Exception as e:
            logger.error("build failed for %s: %s", mst, e)
            if fail_fast:
                raise
    return results

# ----------------------- 메인 (선택) ----------------------
if __name__ == "__main__":
    import argparse, glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--msts', nargs='+', help='List of MST ids to build')
    parser.add_argument('--out-dir', default='faiss_indexes')
    args = parser.parse_args()

    msts = args.msts or []
    if not msts:
        # laws/*.json 목록에서 MST 자동 추출
        for p in glob.glob("laws/*.json"):
            try:
                data = json.load(open(p, encoding="utf-8"))
                law_items = (data.get("LawSearch", {}) or {}).get("law", [])
                if not isinstance(law_items, list):
                    law_items = [law_items]
                for item in law_items:
                    mst = item.get("법령일련번호")
                    if mst and str(mst) not in msts:
                        msts.append(str(mst))
            except Exception:
                continue

    if not msts:
        logger.error("크롤링할 MST가 없습니다. --msts 인자 또는 laws/*.json을 확인하세요.")
        raise SystemExit(1)

    res = build_indexes(msts, args.out_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))

# --- Search helpers: meta-aware rerank (조/항/호) ---
from typing import List, Dict, Any, Optional, Tuple
import os, json
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # noqa

from query_meta import parse_meta
from normalizers import circled_to_int  # 필요 시 사용 (이미 추가한 파일에 있음)

def _load_units_and_idmap(out_dir: str, base: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """base는 JSON 파일 베이스명(예: '안전보건규칙_272927' 또는 '272927')"""
    cand_units = [
        os.path.join(out_dir, f"{base}_units.json"),
        os.path.join(out_dir, base, "units.json"),
        os.path.join(out_dir, "units.json"),
    ]
    cand_idmap = [
        os.path.join(out_dir, f"{base}_faiss_id_map.json"),
        os.path.join(out_dir, base, "faiss_id_map.json"),
        os.path.join(out_dir, "faiss_id_map.json"),
    ]
    units_path = next((p for p in cand_units if os.path.exists(p)), cand_units[0])
    idmap_path = next((p for p in cand_idmap if os.path.exists(p)), cand_idmap[0])
    with open(units_path, "r", encoding="utf-8") as f:
        units = json.load(f)    
    with open(idmap_path, "r", encoding="utf-8") as f:
        idmap = json.load(f)
    return units, idmap

def _collect_meta_matched_ids(units: List[Dict[str,Any]], jo: Optional[str], hang_norm: Optional[int], mok_norm: Optional[int]) -> set:
    """units.json(정규화 필드 포함)에서 조/항/호 일치하는 인덱스(=faiss_id)를 수집"""
    if not jo and not hang_norm and not mok_norm:
        return set()
    matched = []
    for idx, u in enumerate(units):
        if jo and str(u.get("jo_norm") or (u.get("jo") or "")).strip() != str(jo):
            continue
        if hang_norm is not None and u.get("hang_norm") != hang_norm:
            continue
        if mok_norm is not None and u.get("mok_norm") != mok_norm:
            continue
        matched.append(idx)  # id_map의 faiss_id는 units와 동일 인덱스라고 가정
    return set(matched)

class SimpleSearcher:
    """FAISS + 메타 재정렬. 인덱스는 MST명으로 로드(ASCII 경로)."""
    def __init__(self, out_dir: str, mst: str, base: Optional[str] = None):
        """
        out_dir: 인덱스/JSON 산출 폴더
        mst: '272927' 같은 문자열
        base: JSON 베이스명(없으면 mst와 동일한 베이스로 시도)
        """
        self.out_dir = out_dir
        self.mst = str(mst)
        self.base = base or self.mst
        # STRICT_META=1 → 조/항/호 중 2개 이상 일치시에만 '메타 승격'
        self.strict_meta = str(os.environ.get("STRICT_META", "0")).lower() in ("1","true","yes")

        if faiss is None:
            raise RuntimeError("faiss 모듈이 필요합니다.")

        index_path = os.path.join(out_dir, f"{self.mst}_faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        self.index = faiss.read_index(index_path)

        self.units, self.idmap = _load_units_and_idmap(out_dir, self.base)

    def _embed(self, text: str):
        from vector_search_service import embed_text as _embed_text, _sanitize_vec  # 순환 import 회피
        v = _embed_text(_normalize_text(text))
        arr = _sanitize_vec(v)[None, :]  # ✅ 질의 벡터도 살균
        return arr

    def _meta_match_count(self, u: Dict[str, Any], jo: Optional[str],
                          hang_norm: Optional[int], mok_norm: Optional[int]) -> int:
        cnt = 0
        if jo:
            jo_u = u.get("jo_norm") or (u.get("jo") or "").strip()
            if jo_u == jo:
                cnt += 1
        if hang_norm is not None and u.get("hang_norm") == hang_norm:
            cnt += 1
        if mok_norm is not None and u.get("mok_norm") == mok_norm:
            cnt += 1
        return cnt

    def _is_active_as_of(self, u, as_of):
        if not as_of:
            return True
        def _parse(d):
            if not d: return None
            s=str(d).strip()[:10]  # 'YYYY-MM-DD...' 형태일 때 앞 10자
            try:
                import datetime as dt
                y,m,d = s.split('-')
                return dt.date(int(y),int(m),int(d))
            except Exception:
                return None
        cutoff = _parse(as_of)
        if not cutoff:  # 파싱 실패 시 필터 비적용
            return True

        eff = _parse(u.get('effective_date') or u.get('시행일자'))
        amd = _parse(u.get('amended_on') or u.get('개정일자'))
        # 우선순위: 시행일자가 있으면 그 기준, 없으면 개정일자
        keydate = eff or amd
        if not keydate:
            return True
        return keydate <= cutoff
    
    def search(self, query: str, top_k: int = 10, as_of: Optional[str] = None) -> List[Dict[str, Any]]:
        # 1) FAISS 1차 검색(여유 버퍼 포함)
        qv = self._embed(query)
        k = max(top_k * 20, top_k)
        D, I = self.index.search(qv, k)  # D: 거리(작을수록 좋음)
        cand = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            u = self.units[idx]
            if not self._is_active_as_of(u, as_of):
                continue
            cand.append({"faiss_id": idx, "distance": float(dist), "unit": u})

        # 2) 질의에서 조/항/호 파싱 → 매칭 계산
        jo, hang_norm, mok_norm = parse_meta(query)
        meta_counts: Dict[int, int] = {}
        for idx, u in enumerate(self.units):
            if not self._is_active_as_of(u, as_of):
                continue
            c = self._meta_match_count(u, jo, hang_norm, mok_norm)
            if c:
                meta_counts[idx] = c

        # STRICT_META=1 → 2개 이상 일치시에만 메타 승격
        if self.strict_meta:
            require_jo = str(os.environ.get("STRICT_META_REQUIRE_JO", "0")).lower() in ("1","true","yes")
            if require_jo and jo:
                meta_promote = {
                    i for i, c in meta_counts.items()
                    if c >= 2 and (str(self.units[i].get("jo_norm") or (self.units[i].get("jo") or "")).strip() == str(jo))
                }
            else:
                meta_promote = {i for i, c in meta_counts.items() if c >= 2}        
        else:
            meta_promote = _collect_meta_matched_ids(self.units, jo, hang_norm, mok_norm)

        # FAISS 후보에 없는 메타 승격 유닛 주입
        present = {it["faiss_id"] for it in cand}
        for idx in meta_promote:
            if idx not in present:
                u = self.units[idx]
                cand.append({"faiss_id": idx, "distance": 1e9, "unit": u, "injected": True})

        # 3) annex_refs가 있으면 같은 MST의 '별표/서식 annex_no'를 cand에 동반 주입
        #    (보강) FAISS 상위 + 메타 주입(injected=True) + (필요시) 같은 조(jo_norm) 유닛의 annex_refs를 함께 수집
        # 씨드: FAISS 상위
        seed = cand[: max(10, top_k * 2)]
        # 씨드 확장: 메타로 주입된 것들도 포함
        seed += [it for it in cand if it.get("injected") and it not in seed]

        annex_targets: set[str] = set()
        # 3-1) 씨드에서 annex_refs 수집
        for it in seed:
            for r in (it["unit"].get("annex_refs") or []):
                annex_targets.add(str(r))

        # 3-2) 씨드에 annex_refs가 전혀 없고, 질의에서 조(jo)가 파싱되었다면 → 같은 조(jo_norm) 유닛들의 annex_refs 수집
        if not annex_targets:
            jo, hang_norm, mok_norm = parse_meta(query)  # 이미 상단에서 구했다면 재사용해도 됨
            if jo:
                for u1 in self.units:
                    jo_u = str(u1.get("jo_norm") or (u1.get("jo") or "")).strip()
                    if jo_u == jo or (u1.get("jo") and u1.get("jo").strip() == f"제{jo}조"):
                        for r in (u1.get("annex_refs") or []):
                            annex_targets.add(str(r))

        # 3-3) annex_targets 과 일치하는 별표/서식 유닛을 동반 주입
        if annex_targets:
            cand_ids = {it["faiss_id"] for it in cand}
            for idx, u in enumerate(self.units):
                if idx in cand_ids:
                    continue
                if not self._is_active_as_of(u, as_of):
                    continue
                if (u.get("level") in ("별표", "서식")) and (str(u.get("annex_no")) in annex_targets):
                    cand.append({"faiss_id": idx, "distance": 1e9, "unit": u, "annex_injected": True})
        
        # 4) 최종 정렬: 메타 승격 > annex 보너스 > FAISS 거리
        def _prio(item):
            fid = item["faiss_id"]
            pr = 0
            if fid in meta_promote:
                pr += 3
            elif (not self.strict_meta) and (meta_counts.get(fid, 0) >= 1):
                pr += 1
            u = item["unit"]
            # annex 보너스는 메타만큼 강하게 올려준다
            if (u.get("level") in ("별표", "서식")) and (str(u.get("annex_no")) in annex_targets):
                pr += 3
            if item.get("annex_injected"):
                pr += 1
            return (-pr, item["distance"])

        cand.sort(key=_prio)

        # 4-1) 상위 top_k에 annex가 하나도 없으면 1개는 보장 삽입
        selected = cand[:top_k]
        if annex_targets:
            has_annex = any(
                (it["unit"].get("level") in ("별표", "서식")) and
                (str(it["unit"].get("annex_no")) in annex_targets)
                for it in selected
            )
            if not has_annex:
                for it in cand[top_k:]:
                    u = it["unit"]
                    if (u.get("level") in ("별표", "서식")) and (str(u.get("annex_no")) in annex_targets):
                        selected[-1] = it
                        break

         # 5) 상위 top_k 반환(필요 정보만)
        out: List[Dict[str, Any]] = []
        for it in selected:        
            u = it["unit"]
            out.append({
                "faiss_id": it["faiss_id"],
                "distance": it["distance"],
                "level": u.get("level"),
                "annex_no": u.get("annex_no"),
                "jo": u.get("jo"),
                "hang": u.get("hang"),
                "hang_norm": u.get("hang_norm"),
                "mok": u.get("mok"),
                "mok_norm": u.get("mok_norm"),
                "title": u.get("title"),
                "text": u.get("text"),
                "path": u.get("path"),
                "display_path_norm": u.get("display_path_norm"),
            })
        return out
# --- Annex PDF cache ---
_ANNEX_PDF_CACHE_PATH = os.path.join("faiss_indexes", "annex_pdf_cache.json")
_ANNEX_PDF_CACHE: Dict[str, Dict[str, Any]] = {}
_ANNEX_PDF_CACHE_DIRTY = False

def _load_annex_pdf_cache() -> None:
    global _ANNEX_PDF_CACHE
    if _ANNEX_PDF_CACHE:
        return
    try:
        if os.path.exists(_ANNEX_PDF_CACHE_PATH):
            with open(_ANNEX_PDF_CACHE_PATH, "r", encoding="utf-8") as f:
                _ANNEX_PDF_CACHE = json.load(f) or {}
    except Exception:
        _ANNEX_PDF_CACHE = {}

def _save_annex_pdf_cache() -> None:
    global _ANNEX_PDF_CACHE_DIRTY
    if not _ANNEX_PDF_CACHE_DIRTY:
        return
    try:
        os.makedirs(os.path.dirname(_ANNEX_PDF_CACHE_PATH), exist_ok=True)
        with open(_ANNEX_PDF_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_ANNEX_PDF_CACHE, f, ensure_ascii=False, indent=2)
        _ANNEX_PDF_CACHE_DIRTY = False
    except Exception:
        pass
