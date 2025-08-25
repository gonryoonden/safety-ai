# vector_search_service.py
import os
import re
import json
import time
import logging
import tempfile
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

# ----------------------- 임베딩 (플레이스홀더) -------------
import hashlib, struct
def embed_text(text: str) -> List[float]:
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    # 32 bytes -> 8 floats (고정 차원)
    return [struct.unpack('!f', h[i:i+4])[0] for i in range(0, 32, 4)]

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
    law = (law_json.get("법령") if isinstance(law_json, dict) else None) or law_json
    if not isinstance(law, dict):
        return None
    return (law.get("법령약칭명")
            or law.get("법령명_한글")
            or law.get("법령명")
            or None)

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
def extract_units(law_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for art in _get_articles_any_shape(law_json):
        if not isinstance(art, dict):
            continue
        if _is_heading_only(art):
            continue

        jo = _clean(_sg(art, "조문번호") or _sg(art, "조번호") or _sg(art, "조문키"))
        jo_title = _clean(_sg(art, "조문제목"))
        jo_text  = _clean(_sg(art, "조문내용"))
        if jo_title or jo_text:
            units.append({
                "level": "조",
                "jo": jo or None,
                "hang": None,
                "mok": None,
                "title": jo_title or None,
                "text": jo_text or None,
                "path": f"제{jo}조" if jo else "조문",
            })

        for h in _as_list(_sg(art, "항")):
            hang_no  = _clean(_sg(h, "항번호"))
            hang_txt = _clean(_sg(h, "항내용") or _sg(h, "내용"))
            if hang_no or hang_txt:
                units.append({
                    "level": "항",
                    "jo": jo or None,
                    "hang": hang_no or None,
                    "mok": None,
                    "title": None,
                    "text": hang_txt or None,
                    "path": " > ".join([p for p in [f"제{jo}조" if jo else None,
                                                    f"제{hang_no}항" if hang_no else "항"] if p]),
                })

            children = _as_list(_sg(h, "목")) or _as_list(_sg(h, "호"))
            for ch in children:
                num = _clean(_sg(ch, "호번호") or _sg(ch, "목번호") or _sg(ch, "번호"))
                txt = _clean(_sg(ch, "호내용") or _sg(ch, "목내용") or _sg(ch, "내용"))
                if num or txt:
                    units.append({
                        "level": "목",
                        "jo": jo or None,
                        "hang": hang_no or None,
                        "mok": num or None,
                        "title": None,
                        "text": txt or None,
                        "path": " > ".join([p for p in [f"제{jo}조" if jo else None,
                                                        f"제{hang_no}항" if hang_no else None,
                                                        (f"{num}호" if num else None)] if p]),
                    })
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
        index_path   = os.path.join(law_dir, "faiss_index.bin")
    else:
        units_path   = os.path.join(out_dir, f"{base}_units.json")
        answers_path = os.path.join(out_dir, f"{base}_answers.json")
        idmap_path   = os.path.join(out_dir, f"{base}_faiss_id_map.json")
        index_path   = os.path.join(out_dir, f"{base}_faiss_index.bin")

    # 2) 구조화 추출
    units = extract_units(law_json)
    if not units:
        raise RuntimeError(f"MST {mst}: 추출된 조/항/목 단위가 없습니다.")

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
        xb = np.array([embed_text(t) for t in texts_to_embed], dtype='float32')
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        faiss.write_index(index, index_path)
    else:
        with AtomicWriter(index_path) as aw:
            aw.write(b"FAISS_NOT_AVAILABLE")

    dt = time.monotonic() - t0
    logger.info("MST %s build finished in %.2fs (units=%d)", mst, dt, len(units))
    return {"mst": mst, "units": len(units), "duration_sec": dt, "out_dir": out_dir, "base": base}

def build_indexes(msts: List[str], out_dir: str = "faiss_indexes") -> List[Dict[str, Any]]:
    results = []
    oc = os.environ.get("LAW_API_OC")
    shared_client = LawAPIClient(oc) if oc else LawAPIClient()
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
