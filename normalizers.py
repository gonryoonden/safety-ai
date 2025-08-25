# normalizers.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import re
import hashlib

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
    text = u.get("text") or ""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return (jo, hang_norm or 0, mok_norm or 0, title.strip(), text_hash)

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

# -------------------- 메인 후처리 --------------------
def postprocess_units(units: List[Dict[str, Any]], law_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    입력: 원시 units(list of dict)
    출력: 아래 필드들을 보강/정규화 + 보수적 중복 제거
      - hang_norm(int), mok_norm(int)
      - jo_norm(str: '3'|'4의2'), jo_num(int), jo_suffix(int|None)
      - display_path_norm('제n조 제m항 제r호')
      - amended_on(list[str: 'YYYY-MM-DD'])  # 본문 '<개정 ...>'에서 추출
      - (있을 경우) effective_date/promulgation_date/revision_type
    """
    seen = set()
    out: List[Dict[str, Any]] = []

    eff_date = (law_meta or {}).get("시행일자") or (law_meta or {}).get("효력시작일자")
    rev_type = (law_meta or {}).get("제개정구분")
    prom_date = (law_meta or {}).get("공포일자")

    for u in units:
        # 1) 항/호 숫자 정규화
        hang_norm = circled_to_int(u.get("hang"))
        mok_norm = normalize_mok(u.get("mok"))
        u["hang_norm"] = hang_norm
        u["mok_norm"] = mok_norm

        # 2) 조(의조 포함) 표준화
        jo_norm, jo_num, jo_suffix = _normalize_jo_from_fields(
            u.get("jo"),
            u.get("path"),
            u.get("title"),
            u.get("text"),
        )
        if jo_norm:
            u["jo_norm"] = jo_norm
            u["jo_num"] = jo_num
            u["jo_suffix"] = jo_suffix
            u["display_path_norm"] = build_display_path(
                jo_norm or (u.get("jo") or "").strip(),
                u.get("hang_norm"),
                u.get("mok_norm"),
    )
        # 4) 본문에서 개정일 추출
        amends = _extract_amend_dates(u.get("text"))
        if amends:
            u["amended_on"] = amends

        # 5) 법령 메타 주입(있을 때만)
        if eff_date:
            u["effective_date"] = eff_date
        if prom_date:
            u["promulgation_date"] = prom_date
        if rev_type:
            u["revision_type"] = rev_type

        # 6) 보수적 중복 제거(경로/텍스트 동일시 스킵)
        key = unit_stable_key(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(u)

    return out
