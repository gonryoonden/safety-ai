# query_meta.py
import re
from typing import Optional, Tuple

# 원형숫자 ↔ 정수 매핑
CIRCLED_TO_INT = {
    "①":1,"②":2,"③":3,"④":4,"⑤":5,"⑥":6,"⑦":7,"⑧":8,"⑨":9,"⑩":10,
    "⑪":11,"⑫":12,"⑬":13,"⑭":14,"⑮":15,"⑯":16,"⑰":17,"⑱":18,"⑲":19,"⑳":20,
}

def _to_int_hang(token: str) -> Optional[int]:
    token = (token or "").strip()
    if token in CIRCLED_TO_INT:
        return CIRCLED_TO_INT[token]
    m = re.search(r"(\d+)", token)
    return int(m.group(1)) if m else None

def parse_meta(q: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """질의에서 제n조(의m), 제k항, 제r호 추출.
    return: (jo_str, hang_norm, mok_norm)
      - jo_str: "4" 또는 "4의2" (의조 포함)
      - hang_norm, mok_norm: 정수 or None
    """
    q = (q or "").strip()

    # 제n조(의m)  ex) 제4조의2, 4조의2, 제 12 조
    jo = None
    m = re.search(r"(?:제\s*)?(\d+)\s*조\s*(?:의\s*(\d+))?", q)
    if m:
        jo_num = m.group(1)
        jo_suf = m.group(2)
        jo = f"{jo_num}의{jo_suf}" if jo_suf else jo_num

    # 제k항 (원형숫자 포함)  ex) 제3항, ③항
    hang_norm = None
    m = re.search(r"(?:제\s*)?([0-9①-⑳]+)\s*항", q)
    if m:
        hang_norm = _to_int_hang(m.group(1))

    # 제r호  ex) 제2호, 2호
    mok_norm = None
    m = re.search(r"(?:제\s*)?(\d+)\s*호", q)
    if m:
        mok_norm = int(m.group(1))

    return jo, hang_norm, mok_norm
