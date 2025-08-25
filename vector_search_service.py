# vector_search_service.py
import os
import re
import json
import time
import logging
import tempfile
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple
from normalizers import postprocess_units


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
import numpy as np
from typing import List

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
    """
    법령명(한글)을 최대한 보수적으로 찾아 반환.
    - 1차: {"법령": {...}} 바로 아래에서 표준 키 조회
    - 2차: 중첩 전체를 깊이 우선으로 탐색
    - 3차: 키 변형(언더스코어 없는 "법령명한글" 등)까지 허용
    """
    if not isinstance(law_json, dict):
        return None
    law = law_json.get("법령") if isinstance(law_json.get("법령"), dict) else law_json

    # 후보 키들(우선순위대로)
    PREFERRED_KEYS = [
        "법령약칭명", "법령명_한글", "법령명",
        "법령명한글", "한글법령명", "법령한글명",
    ]

    # 1) 얕은 검색
    for k in PREFERRED_KEYS:
        v = law.get(k) if isinstance(law, dict) else None
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()

    # 2) 깊은 탐색
    def walk(obj):
        if isinstance(obj, dict):
            # 우선 표준 키들
            for k in PREFERRED_KEYS:
                v = obj.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    return str(v).strip()
            # 그 다음 전체 키 스캔
            for k, v in obj.items():
                # 키 정규화(언더스코어 제거/소문자)
                kk = re.sub(r"[_\s]+", "", str(k)).lower()
                if kk in ("법령명한글", "한글법령명", "법령한글명"):
                    if isinstance(v, (str, int)) and str(v).strip():
                        return str(v).strip()
                res = walk(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for it in obj:
                res = walk(it)
                if res:
                    return res
        return None

    return walk(law)

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

    articles = _get_articles_any_shape(law_json)  # 다양한 형태의 '조문'을 평탄화
    for art in articles:
        if _is_heading_only(art):  # 편/장/절/관/부칙/별표/서식 등 헤딩만인 경우 스킵
            continue

        jo_raw   = _clean(_sg(art, "조문번호"))
        jo_title = _clean(_sg(art, "조문제목"))
        jo_body  = _clean(_stringify(_sg(art, "조문내용")))

        # 1) 조 단위
        if jo_raw and (jo_title or jo_body):
            units.append({
                "level": "조",
                "jo": jo_raw, "hang": None, "mok": None,
                "title": jo_title or None,
                "text": jo_body or "",
                "path": f"제{jo_raw}조",
            })

        # 2) 항 단위
        for h in _as_list(_sg(art, "항")):
            hang_no   = _clean(_sg(h, "항번호") or _sg(h, "항"))
            hang_body = _clean(_stringify(_sg(h, "항내용") or h))
            if not hang_no or not hang_body:
                continue

            h_path = f"제{jo_raw}조 > 제{hang_no}항" if jo_raw else f"제{hang_no}항"
            units.append({
                "level": "항",
                "jo": jo_raw, "hang": hang_no, "mok": None,
                "title": None,
                "text": hang_body,
                "path": h_path,
            })

            # 3) 목(또는 호) 단위
            m_list = _as_list(_sg(h, "목") or _sg(h, "호"))
            for m in m_list:
                mok_no   = _clean(_sg(m, "목번호") or _sg(m, "호번호") or _sg(m, "목") or _sg(m, "호"))
                mok_body = _clean(_stringify(_sg(m, "목내용") or _sg(m, "호내용") or m))
                if not mok_no or not mok_body:
                    continue

                # 기존 데이터 스타일을 따라 "1.호" 형태로 맞춤
                _seg = mok_no.strip()
                seg = f"{_seg}호" if _seg.endswith(".") else f"{_seg}.호"

                units.append({
                    "level": "목",
                    "jo": jo_raw, "hang": hang_no, "mok": mok_no,
                    "title": None,
                    "text": mok_body,
                    "path": f"{h_path} > {seg}",
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
            })
    except Exception:
        return units
    return units


def fetch_annex_units(client: LawAPIClient, mst: str, law_title: Optional[str]) -> List[Dict[str, Any]]:
    """
    licbyl(별표/서식) 검색 → 해당 MST만 수집해 유닛화.
    최소 버전: 제목/번호/링크만 채움 (본문 텍스트는 제목 복제)
    """
    units: List[Dict[str, Any]] = []

    def _rows(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """응답 어디에 있든 licbyl을 리스트로 반환."""
        if not isinstance(resp, dict):
            return []
        blk = resp.get("licBylSearch") or resp.get("LicBylSearch") or resp.get("licbylsearch")
        rows = (blk.get("licbyl") if isinstance(blk, dict) else None) or resp.get("licbyl")
        if rows:
            return rows if isinstance(rows, list) else [rows]
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
    
     # 1차: law_title(법령명) 기반 '해당법령검색(search=2)'
    rows = []
    if law_title:
        try:
            logger.info(f"[annex] licbyl search=2 by law_title={law_title!r}")
            resp = client.search_attachments(query=str(law_title), search=2, display=100)
            rows = _rows(resp)
            logger.info(f"[annex] search=2 rows={len(rows)}")
        except Exception:
            rows = []

    # 2차: law_title로 '별표/서식명(search=1)'
    if not rows and law_title:
        try:
            logger.info(f"[annex] licbyl search=1 by law_title={law_title!r}")
            resp2 = client.search_attachments(query=str(law_title), search=1, display=100)
            rows = _rows(resp2)
            logger.info(f"[annex] search=1 rows={len(rows)}")
        except Exception:
            rows = []

    # 3차: law_title로 '본문검색(search=3)'
    if not rows and law_title:
        try:
            logger.info(f"[annex] licbyl search=3 by law_title={law_title!r}")
            resp3 = client.search_attachments(query=str(law_title), search=3, display=100)
            rows = _rows(resp3)
            logger.info(f"[annex] search=3 rows={len(rows)}")
        except Exception:
            rows = []

    # (옵션) 마지막 폴백: mst 문자열로 search=2
    if not rows:
        try:
            logger.info(f"[annex] licbyl search=2 by mst={mst}")
            resp4 = client.search_attachments(query=str(mst), search=2, display=100)
            rows = _rows(resp4)
            logger.info(f"[annex] search=2(mst) rows={len(rows)}")
        except Exception:
            rows = []

    if not rows:
        return units

    LAW_BASE = os.environ.get("LAW_BASE", "http://www.law.go.kr")

    def _norm_annex_no(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = str(s).strip()
        # "4-2" / "4/2" / "4 의 2" / "4의2" 등 → "4의2"
        m = re.match(r'^(\d+)\s*(?:[-_/]|의)\s*(\d+)$', s)
        if m:
            return f"{m.group(1)}의{m.group(2)}"
        # 단일 숫자는 그대로
        m2 = re.match(r'^\d+$', s)
        if m2:
            return s
        return s

    def _get_any(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for k in keys:
            v = _sg(d, k)
            if v:
                return v
        return None

    def _abs(u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        return u if str(u).startswith("http") else (LAW_BASE + str(u))

    for r in rows:
        # 관련 법령 일련번호가 있으면 mst로 필터, 없으면 통과
        rel = _clean(_get_any(r, ["관련법령일련번호", "법령일련번호", "MST"]))
        if rel and rel != str(mst):
            continue

        name = _clean(_get_any(r, ["별표명", "서식명", "항목명", "명칭", "제목"]))
        annex_no = _norm_annex_no(_get_any(r, ["별표번호", "번호"]))

        html = _get_any(r, ["별표서식파일링크","별표본문링크","서식파일링크","파일링크","본문링크"])
        pdf  = _get_any(r, ["별표서식PDF파일링크","PDF파일링크","서식PDF링크"])
        det  = _get_any(r, ["별표법령상세링크","상세링크","법령상세링크"])
        links = {}
        if html: links["html"]   = _abs(html)
        if pdf:  links["pdf"]    = _abs(pdf)
        if det:  links["detail"] = _abs(det)
        # level 판정: '서식명' 키가 있으면 서식, 아니면 별표
        is_form = _get_any(r, ["서식명"]) is not None
        level = "서식" if is_form else "별표"
        path_label = f"{level} {annex_no}" if annex_no else level

        units.append({
            "level": level,
            "jo": None, "hang": None, "mok": None,
            "title": name or "별표/서식",
            "text": name or "별표/서식",                         # 최소 버전: 본문 = 제목
            "path": path_label,   # 표시 경로
            "annex_no": annex_no,
            "links": links or None,
        })

         # --- annex 중복 정리: 같은 annex_no는 '최고본'만 유지 ---
    def _score(u: Dict[str, Any]) -> int:
        L = u.get("links") or {}
        s = 0
        if L.get("html"):   s += 4
        if L.get("pdf"):    s += 2
        if L.get("detail"): s += 1
        title = (u.get("title") or "").strip()
        if title: s += min(50, len(title))
        # 별표를 서식보다 우선시하려면 가중치 (선택)
        if u.get("level") == "별표": s += 1
        return s

    by_no: Dict[str, Dict[str, Any]] = {}
    for u in units:
        key = str(u.get("annex_no") or (u.get("title") or ""))
        if key not in by_no or _score(u) > _score(by_no[key]):
            by_no[key] = u
    units = list(by_no.values())

    return units

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

# ----------------------- 인덱스 빌드 ----------------------
def build_index_for_mst(mst: str, out_dir: str = "faiss_indexes", client: Optional[LawAPIClient] = None) -> Dict[str, Any]:
    client = client or LawAPIClient()
    t0 = time.monotonic()

    # 0) 수집
    law_json = fetch_full_law(client, mst)

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

    annexes = fetch_annex_units(client, mst, law_title)
    if annexes:
        units.extend(annexes)
        logger.info(f"MST {mst}: 별표/서식 units += {len(annexes)}")

    # 2.3) ✅ 후처리(한 번에)
    units = postprocess_units(units, law_meta=law_json.get("법령"))

    # 3) units 저장
    with AtomicWriter(units_path) as aw:
        aw.write(json.dumps(units, ensure_ascii=False, indent=2).encode('utf-8'))

    # 4) 임베딩 입력 & id_map
    texts_to_embed: List[str] = []
    id_map: List[Dict[str, Any]] = []
    for i, u in enumerate(units):
        combined = ((u.get("title") or "") + "\n" if u.get("title") else "") + (u.get("text") or "")
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
        import numpy as np
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
def build_indexes(msts: List[str], out_dir: str = "faiss_indexes") -> List[Dict[str, Any]]:
    results = []
    oc = os.environ.get("LAW_API_OC")
    shared_client = LawAPIClient()
    for mst in msts:
        try:
            results.append(build_index_for_mst(mst, out_dir, client=shared_client))
        except Exception as e:
            logger.error("build failed for %s: %s", mst, e)
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
        v = _embed_text(text)
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
