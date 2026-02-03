# normalizers.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import re
import hashlib
import unicodedata
import time
from urllib.parse import quote

def _is_api_url(url: Optional[str]) -> bool:
    if not url:
        return False
    s = str(url)
    return ("/DRF/lawService.do" in s) or ("OC=" in s) or ("/DRF/" in s)

# -------------------- 숫자/표기 정규화 테이블 --------------------
_CIRCLED_TO_INT = {
    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
    "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
    "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15,
    "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
}
_INT_TO_CIRCLED = {v: k for k, v in _CIRCLED_TO_INT.items()}

# 의조/조 인식용: "제4조", "제4조의2", "4", "4의2" 등
_JO_RE = re.compile(r'(?:제\s*)?(\d+)\s*조(?:\s*의\s*(\d+))?', re.UNICODE)

# 본문 내 개정일 추출: "<개정 2012.3.5, 2019.12.26>" 같은 패턴 지원
_AMEND_TAG_RE = re.compile(r'<\s*개정[^>]*>', re.UNICODE)
_DATE_RE = re.compile(r'(\d{4})\.(\d{1,2})\.(\d{1,2})')
# 본문 내 '별표 n(의m)' 참조 추출용
_ANNEX_REF_RE = re.compile(r"별표\s*(\d+)(?:\s*의\s*(\d+))?")

# -------------------- 기본 정규화 함수 --------------------
def circled_to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if s in _CIRCLED_TO_INT:
        return _CIRCLED_TO_INT[s]
    m = re.search(r'(\d+)', s)  # "제3항", "3", "3." 등 대응
    return int(m.group(1)) if m else None

def normalize_mok(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    m = re.match(r'^\s*(\d+)\s*\.?\s*$', s)  # "2." / "2" → 2
    if m:
        return int(m.group(1))
    return None  # (가/나 등은 필요 시 추가)

def build_display_path(jo_like: Optional[str], hang_norm: Optional[int], mok_norm: Optional[int]) -> str:
    """
    jo_like: "4" 또는 "4의2" (jo_norm 권장)
    출력 예: "제4조", "제4조의2", "제4조의2 제1항", "제4조의2 제1항 제2호"
    """
    parts = []
    if jo_like:
        m = re.match(r'^\s*(\d+)(?:\s*의\s*(\d+))?\s*$', str(jo_like))
        if m:
            base, ext = m.group(1), m.group(2)
            head = f"제{int(base)}조"
            if ext:
                head += f"의{int(ext)}"
            parts.append(head)
        else:
            # 그래도 뭔가 들어왔으면 그대로 표시(최후의 안전장치)
            parts.append(str(jo_like))

    if hang_norm:
        parts.append(f"제{int(hang_norm)}항")
    if mok_norm:
        parts.append(f"제{int(mok_norm)}호")

    return " ".join(parts)

def unit_stable_key(u: Dict[str, Any]) -> Tuple:
    jo = u.get("jo_norm") or u.get("jo")
    hang_norm = u.get("hang_norm")
    mok_norm = u.get("mok_norm")
    title = u.get("title") or ""
    item_type = u.get("item_type") or ""
    item_path = u.get("item_path") or u.get("item_path_raw") or ""
    text = u.get("text") or ""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return (jo, hang_norm or 0, mok_norm or 0, item_type, item_path, title.strip(), text_hash)

# -------------------- 신규: jo(의조 포함) 표준화 & 개정일 추출 --------------------
def _normalize_jo_from_fields(jo_val, path_val, title_val=None, text_val=None):
    """
    jo_norm 우선순위:
      1) '의조'가 포함된 매치(예: 제4조의2)를 최우선
      2) 없다면 첫 매치 사용
    """
    best = None
    for s in (jo_val, path_val, title_val, text_val):
        if not s:
            continue
        m = _JO_RE.search(str(s).strip())
        if not m:
            continue
        base, ext = m.group(1), m.group(2)
        jo_norm = f"{base}의{ext}" if ext else base
        jo_num = int(base)
        jo_suffix = int(ext) if ext else None
        # 의조(=ext 있음) 발견 즉시 반환
        if ext:
            return jo_norm, jo_num, jo_suffix
        # 의조가 없으면 일단 후보로 저장
        if best is None:
            best = (jo_norm, jo_num, jo_suffix)
    return best if best is not None else (None, None, None)

def _extract_amend_dates(text: Optional[str]) -> List[str]:
    """본문 문자열에서 <개정 ...> 태그 내 YYYY.M.D 패턴들을 모두 ISO(YYYY-MM-DD)로 반환"""
    if not text:
        return []
    out: List[str] = []
    for tag in _AMEND_TAG_RE.findall(text):
        for y, m, d in _DATE_RE.findall(tag):
            iso = f"{y}-{int(m):02d}-{int(d):02d}"
            if iso not in out:
                out.append(iso)
    return out

def _extract_law_title(law_meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(law_meta, dict):
        return None
    for k in ("법령명", "법령명한글", "법령명_한글", "법령약칭"):
        v = law_meta.get(k)
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
    return walk(law_meta)


def _extract_law_mst(law_meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(law_meta, dict):
        return None
    for k in ("법령일련번호", "MST"):
        v = law_meta.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return None


def _extract_law_source_url(law_meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(law_meta, dict):
        return None
    v = law_meta.get("법령상세링크")
    if isinstance(v, (str, int)) and str(v).strip():
        return str(v).strip()
    return None


def _normalize_text(text: Optional[str]) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    # unify quotes/brackets
    repl = {
        "“": "\"", "”": "\"", "‘": "'", "’": "'",
        "「": "\"", "」": "\"", "『": "\"", "』": "\"",
        "【": "[", "】": "]", "〔": "[", "〕": "]",
        "（": "(", "）": ")", "〈": "(", "〉": ")",
        "《": "(", "》": ")", "［": "[", "］": "]",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    # unit normalization (simple)
    s = re.sub(r"(\d)\s*미터", r"\1m", s)
    s = re.sub(r"(\d)\s*m\b", r"\1m", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_item_token(token: Optional[str]) -> str:
    if token is None:
        return ""
    s = unicodedata.normalize("NFKC", str(token))
    s = s.strip()
    s = s.strip("()[]{}<>【】〔〕「」『』")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[\.．。ㆍ,]+$", "", s)
    if s.startswith("제"):
        s = s[1:]
    for suf in ("세목", "목", "호", "항", "조"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    # circled numbers
    c = circled_to_int(s)
    if c is not None and str(c) != s:
        s = str(c)
    m = re.match(r"^(\d+)\s*의\s*(\d+)$", s)
    if m:
        return f"{int(m.group(1))}의{int(m.group(2))}"
    m = re.match(r"^(\d+)$", s)
    if m:
        return str(int(m.group(1)))
    return s

def _normalize_item_path(raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None
    parts = re.split(r"\s*-\s*", str(raw_path).strip())
    norm = [_normalize_item_token(p) for p in parts if p and _normalize_item_token(p)]
    return "-".join(norm) if norm else None

def _format_article_label(jo_norm: Optional[str], jo_raw: Optional[str]) -> Optional[str]:
    base = (jo_norm or (str(jo_raw).strip() if jo_raw else None))
    if not base:
        return None
    m = re.match(r"^(\d+)(?:의(\d+))?$", base)
    if m:
        if m.group(2):
            return f"제{int(m.group(1))}조의{int(m.group(2))}"
        return f"제{int(m.group(1))}조"
    if str(base).startswith("제") and "조" in str(base):
        return str(base)
    return f"제{base}조"

def _format_paragraph_label(hang_norm: Optional[int], hang_raw: Optional[str]) -> Optional[str]:
    if hang_norm:
        return f"제{int(hang_norm)}항"
    if hang_raw:
        s = str(hang_raw).strip()
        if s.startswith("제") and s.endswith("항"):
            return s
        return f"제{s}항"
    return None

def _annex_no_to_human(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    s = str(code).strip()
    if not s:
        return None
    if s.isdigit() and len(s) == 6:
        a = int(s[:4]); b = int(s[4:])
        return f"{a}의{b}" if b else str(a)
    if s.isdigit():
        return str(int(s))
    return s

def _normalize_annex_human(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = str(s).strip()
    t = re.sub(r"^(별표|서식)\s*", "", t)
    t = t.strip()
    m = re.match(r"^(\d+)(?:\s*의\s*(\d+))?$", t)
    if m:
        a = int(m.group(1)); b = int(m.group(2) or 0)
        return f"{a}의{b}" if b else str(a)
    return t

def _annex_human_to_code(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = _normalize_annex_human(s)
    if not t:
        return None
    m = re.match(r"^(\d+)(?:\s*의\s*(\d+))?$", t)
    if not m:
        return None
    a = int(m.group(1)); b = int(m.group(2) or 0)
    return f"{a:04d}{b:02d}"

def _detect_tags(text: Optional[str]) -> List[str]:
    s = _normalize_text(text or "")
    tags: List[str] = []
    def has_any(keys: List[str]) -> bool:
        return any(k in s for k in keys)
    if has_any(["다만", "제외", "그러하지 아니", "아니한다", "아니한", "아니할", "아니하면"]):
        tags.append("EXCEPTION")
    if has_any(["라 한다", "이라 한다", "란", "이라 함"]):
        tags.append("DEFINITION")
    if has_any(["벌칙", "과태료", "처벌", "벌금", "징역", "금고", "자격정지", "형벌"]):
        tags.append("PENALTY")
    if has_any(["할 수 있다", "할수있다", "할 수 있는", "하지 아니할 수 있다", "할 수 있음"]):
        tags.append("DISCRETION")
    return tags

# -------------------- 메인 후처리 --------------------

def postprocess_units(
    units: List[Dict[str, Any]],
    law_meta: Optional[Dict[str, Any]] = None,
    mst: Optional[str] = None,
    source_url: Optional[str] = None,
    crawl_ts: Optional[str] = None,
    law_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    입력: 원시 units(list of dict)
    출력: 아래 필드들을 보강/정규화 + 보수적 중복 제거
      - hang_norm(int), mok_norm(int)
      - jo_norm(str: '3'|'4의2'), jo_num(int), jo_suffix(int|None)
      - display_path_norm('제n조 제m항 제r호')  # 조/항/목일 때
      - amended_on(list[str: 'YYYY-MM-DD'])    # 본문 '<개정 ...>'에서 추출
      - (있을 경우) effective_date/promulgation_date/revision_type
    """
    seen = set()
    out: List[Dict[str, Any]] = []

    law_title = law_title or _extract_law_title(law_meta)
    mst_val = (str(mst).strip() if mst else None) or _extract_law_mst(law_meta)
    law_url = source_url or _extract_law_source_url(law_meta)
    if law_url and _is_api_url(law_url):
        law_url = None
    if not law_url and law_title:
        try:
            law_enc = quote(str(law_title))
            law_url = f"https://www.law.go.kr/법령/{law_enc}"
        except Exception:
            law_url = None
    if crawl_ts is None:
        crawl_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    eff_date = (law_meta or {}).get("시행일자") or (law_meta or {}).get("시행시작일자")
    rev_type = (law_meta or {}).get("개정구분")
    prom_date = (law_meta or {}).get("공포일자")

    for u in units:
        level = u.get("level")

        # 0) 법령 기본 메타 주입
        if law_title:
            u["law"] = law_title
            if not u.get("law_title"):
                u["law_title"] = law_title
        if mst_val:
            u["mst"] = mst_val

        # 1) 항/호 숫자 정규화
        hang_norm = circled_to_int(u.get("hang"))
        mok_norm  = normalize_mok(u.get("mok"))
        u["hang_norm"] = hang_norm
        u["mok_norm"]  = mok_norm

        # 2) 조(의조 포함) 표준화
        jo_norm, jo_num, jo_suffix = _normalize_jo_from_fields(
            u.get("jo"), u.get("path"), u.get("title"), u.get("text")
        )
        if jo_norm:
            u["jo_norm"]   = jo_norm
            u["jo_num"]    = jo_num
            u["jo_suffix"] = jo_suffix

        # 2-1) 조/항/목 메타데이터
        if level in ("조", "항", "목"):
            article_label = _format_article_label(jo_norm, u.get("jo"))
            if article_label:
                u["article"] = article_label
            if level == "조" and u.get("title"):
                u["article_title"] = u.get("title")
            paragraph_label = _format_paragraph_label(hang_norm, u.get("hang"))
            if paragraph_label:
                u["paragraph"] = paragraph_label
            raw_item_path = u.get("item_path_raw") or u.get("item_path") or u.get("mok")
            item_path = _normalize_item_path(raw_item_path)
            if item_path and level == "목":
                u["item_path"] = item_path
            if level == "목" and not u.get("item_type"):
                # 호/목 혼종 폴백: 숫자면 호로 추정
                if mok_norm is not None:
                    u["item_type"] = "호"
                else:
                    u["item_type"] = "목"
            if not u.get("article"):
                # display_path_norm에서 조 추출 폴백
                dp = u.get("display_path_norm") or u.get("path") or ""
                m = re.search(r"(제\\s*\\d+\\s*조(?:\\s*의\\s*\\d+)?)", dp)
                if m:
                    u["article"] = re.sub(r"\\s+", "", m.group(1))
        else:
            if not u.get("article"):
                u["article"] = u.get("display_path_norm") or u.get("title") or level or None

        # 3) 표시 경로
        if level in ("조", "항", "목"):
            u["display_path_norm"] = build_display_path(
                jo_norm or (u.get("jo") or "").strip(),
                hang_norm, mok_norm
            )
        else:
            # 별표/부칙 등은 path 그대로 사용(없으면 title/level 폴백)
            u["display_path_norm"] = u.get("path") or (u.get("title") or level or "")

        if level in ("별표", "서식"):
            human = u.get("annex_no_human") or _annex_no_to_human(u.get("annex_no"))
            human = _normalize_annex_human(human)
            if human:
                u["annex_no_human"] = human

        # 4) 본문에서 개정일 추출
        amends = _extract_amend_dates(u.get("text"))
        if amends:
            u["amended_on"] = amends

        # 5) 본문 내 '별표 n(의m)' 참조 → annex_refs
        if level in ("조", "항", "목"):
            refs = []
            for m in _ANNEX_REF_RE.finditer(u.get("text") or ""):
                base, ext = m.group(1), m.group(2)
                refs.append(f"{int(base)}의{int(ext)}" if ext else str(int(base)))
            if refs:
                u["annex_refs"] = sorted(set(refs), key=lambda x: (len(x), x))
                u["refs"] = [f"별표{r}" for r in u["annex_refs"]]

        # 5-1) 정규화 텍스트 + 태그
        norm_text = _normalize_text(u.get("text"))
        if norm_text:
            u["normalized_text"] = norm_text
        tags = set(u.get("tags") or [])
        tags.update(_detect_tags(u.get("text") or ""))
        if tags:
            u["tags"] = sorted(tags)

        # 6) 법령 메타 주입(있을 때만)
        if eff_date:
            u["effective_date"] = eff_date
        if prom_date:
            u["promulgation_date"] = prom_date
        if rev_type:
            u["revision_type"] = rev_type
        if not u.get("effective_date") and not u.get("as_of"):
            # fallback to promulgation date or crawl date (YYYY-MM-DD)
            if prom_date:
                u["as_of"] = prom_date
            elif crawl_ts:
                u["as_of"] = str(crawl_ts)[:10]

        # 6-1) source_url ???????????? ??? ???)
        if u.get("source_url"):
            su = str(u.get("source_url")).strip()
            if _is_api_url(su):
                u.pop("source_url", None)
        if not u.get("source_url"):
            links = u.get("links") or {}
            if isinstance(links, dict):
                for k in ("detail", "html", "pdf"):
                    if links.get(k):
                        url = str(links.get(k))
                        if url.startswith("/"):
                            url = "https://www.law.go.kr" + url
                        if _is_api_url(url):
                            continue
                        u["source_url"] = url
                        break
            if not u.get("source_url") and law_url:
                if not _is_api_url(law_url):
                    u["source_url"] = law_url

        # 6-1b) source_url_abs / source_url_type
        source_url = u.get("source_url")
        abs_url = None
        if source_url:
            s = str(source_url).strip()
            if s.startswith("/"):
                abs_url = "https://www.law.go.kr" + s
            else:
                abs_url = s
        # avoid API URLs in source_url_abs (DRF/lawService, OC=)
        if abs_url and _is_api_url(abs_url):
            abs_url = None
        if not abs_url:
            lvl = u.get("level")
            art = u.get("article")
            if lvl in ("조", "항", "목") and law_title and art:
                try:
                    from urllib.parse import quote
                    law_enc = quote(str(law_title))
                    art_num = str(art).replace("조", "").replace("제", "")
                    abs_url = f"https://www.law.go.kr/법령/{law_enc}/제{art_num}조"
                    u["source_url_type"] = "article"
                except Exception:
                    abs_url = None
            if not abs_url and law_title:
                try:
                    from urllib.parse import quote
                    law_enc = quote(str(law_title))
                    abs_url = f"https://www.law.go.kr/법령/{law_enc}"
                    u["source_url_type"] = "law_full"
                except Exception:
                    abs_url = None
            if not abs_url and law_url:
                lu = str(law_url).strip()
                if _is_api_url(lu):
                    lu = None
                if lu:
                    abs_url = lu
                    u["source_url_type"] = "law_full"
        else:
            lvl = u.get("level")
            if lvl in ("별표", "서식"):
                u["source_url_type"] = "annex"
            elif u.get("source_url_type") is None:
                u["source_url_type"] = "unknown"

        if abs_url:
            if not u.get("source_url"):
                u["source_url"] = abs_url
            u["source_url_abs"] = abs_url

        if (not u.get("source_url")) and law_url and (not _is_api_url(law_url)):
            u["source_url"] = law_url
        if (not u.get("source_url_abs")) and law_url and (not _is_api_url(law_url)):
            u["source_url_abs"] = law_url

        if not u.get("source_url_abs") and u.get("law"):
            try:
                from urllib.parse import quote
                law_enc2 = quote(str(u.get("law")))
                u["source_url_abs"] = f"https://www.law.go.kr/법령/{law_enc2}"
            except Exception:
                pass
        if not u.get("source_url") and u.get("source_url_abs"):
            u["source_url"] = u.get("source_url_abs")

        # 6-2) 스냅샷 해시/크롤 시각 + source_anchor
        text = u.get("text") or ""
        snapshot_hash = u.get("snapshot_hash") or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        u["snapshot_hash"] = snapshot_hash
        if crawl_ts and not u.get("crawl_ts"):
            u["crawl_ts"] = crawl_ts
        if not u.get("source_anchor"):
            anchor = None
            annex_id = u.get("annex_id")
            if annex_id:
                anchor = f"annex:{annex_id}"
            elif u.get("annex_no"):
                anchor = f"annex_no:{u.get('annex_no')}"
            else:
                parts = []
                if mst_val:
                    parts.append(f"mst:{mst_val}")
                if u.get("article_id"):
                    parts.append(f"jo:{u.get('article_id')}")
                if u.get("paragraph_id"):
                    parts.append(f"hang:{u.get('paragraph_id')}")
                if u.get("item_id"):
                    parts.append(f"item:{u.get('item_id')}")
                if parts and (len(parts) > (1 if mst_val else 0)):
                    anchor = "|".join(parts)
            if not anchor:
                # 구조적 키(jo_norm/hang_norm/item_path)로 2차 폴백
                parts2 = []
                if mst_val:
                    parts2.append(f"mst:{mst_val}")
                if u.get("jo_norm"):
                    parts2.append(f"jo:{u.get('jo_norm')}")
                if u.get("hang_norm"):
                    parts2.append(f"hang:{u.get('hang_norm')}")
                if u.get("item_path"):
                    parts2.append(f"item:{u.get('item_path')}")
                if parts2 and (len(parts2) > (1 if mst_val else 0)):
                    anchor = "|".join(parts2)
            if not anchor:
                base = mst_val or (law_title or "")
                anchor = f"fallback:{base}:{level or ''}:{snapshot_hash}:{crawl_ts}"
            u["source_anchor"] = anchor

        # 7) 보수적 중복 제거(경로/텍스트 동일시 스킵)
        key = unit_stable_key(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(u)

    # 8) 별표 역참조(ref_by) 생성 (누락 별표는 플레이스홀더 생성)
    annex_index: Dict[str, List[Dict[str, Any]]] = {}
    for u in out:
        if u.get("level") in ("별표", "서식"):
            human = u.get("annex_no_human") or _annex_no_to_human(u.get("annex_no"))
            human = _normalize_annex_human(human)
            if human:
                annex_index.setdefault(human, []).append(u)

    # 플레이스홀더 생성 (본문 참조는 있으나 별표 유닛이 없는 경우)
    missing_refs_all: set = set()
    for u in out:
        for r in (u.get("annex_refs") or []):
            if r not in annex_index:
                missing_refs_all.add(r)
    for r in sorted(missing_refs_all):
        code = _annex_human_to_code(r)
        placeholder_url = None
        if law_url and (not _is_api_url(law_url)):
            placeholder_url = law_url
        elif law_title:
            try:
                from urllib.parse import quote
                placeholder_url = f"https://www.law.go.kr/법령/{quote(str(law_title))}"
            except Exception:
                placeholder_url = None
        placeholder = {
            "level": "별표",
            "annex_no": code,
            "annex_no_human": r,
            "title": f"별표 {r} (미수집)",
            "text": "",
            "law": law_title,
            "mst": mst_val,
            "source_url": placeholder_url,
            "source_url_abs": placeholder_url,
            "source_anchor": f"annex_no:{code or r}",
            "annex_placeholder": True,
        }
        out.append(placeholder)
        annex_index.setdefault(r, []).append(placeholder)

    if annex_index:
        for u in out:
            refs = u.get("annex_refs") or []
            if not refs:
                continue
            missing: List[str] = []
            for r in refs:
                targets = annex_index.get(r)
                if not targets:
                    missing.append(r)
                    continue
                for t in targets:
                    ref_by = t.setdefault("ref_by", [])
                    anchor = u.get("source_anchor")
                    if anchor and anchor not in ref_by:
                        ref_by.append(anchor)
                    # 선택적: 사람이 읽기 쉬운 경로
                    disp = u.get("display_path_norm") or u.get("article")
                    if disp:
                        ref_by_disp = t.setdefault("ref_by_display", [])
                        if disp not in ref_by_disp:
                            ref_by_disp.append(disp)
            if missing:
                u["annex_refs_missing"] = missing

    return out
