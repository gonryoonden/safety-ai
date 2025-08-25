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
    # ✅ 후처리: 항/호 정규화, display_path_norm, (메타 주입/중복 보수적 제거)
    units = postprocess_units(units, law_meta=law_json.get("법령"))

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
    units_path = os.path.join(out_dir, f"{base}_units.json") if os.path.exists(os.path.join(out_dir, f"{base}_units.json")) \
        else os.path.join(out_dir, "units.json")  # BUNDLE_PER_LAW 대비
    idmap_path = os.path.join(out_dir, f"{base}_faiss_id_map.json") if os.path.exists(os.path.join(out_dir, f"{base}_faiss_id_map.json")) \
        else os.path.join(out_dir, "faiss_id_map.json")
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
        if jo and u.get("jo") != jo:
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

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        import numpy as np
        # 1) FAISS 1차 검색(여유 버퍼 포함)
        qv = self._embed(query)
        k = max(top_k * 20, top_k)
        D, I = self.index.search(qv, k)  # D: 거리(작을수록 좋음) 가정
        cand = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            u = self.units[idx]
            cand.append({"faiss_id": idx, "distance": float(dist), "unit": u})

        # 2) 질의에서 조/항/호 파싱 → 매칭 ID 집합
        jo, hang_norm, mok_norm = parse_meta(query)
        matched_ids = _collect_meta_matched_ids(self.units, jo, hang_norm, mok_norm)

        # 3) 메타 우선 재정렬 (우선순위 → 원래 거리)
        def _prio(item):
            fid = item["faiss_id"]
            pr = 0
            if fid in matched_ids:
                # 조/항/호 모두 매칭 시 가중치 ↑
                pr = 2
                u = item["unit"]
                if (hang_norm is None or u.get("hang_norm") == hang_norm) and (mok_norm is None or u.get("mok_norm") == mok_norm):
                    pr = 3
            return (-pr, item["distance"])

        cand.sort(key=_prio)
        # 4) 상위 top_k 반환(필요 정보만)
        out = []
        for it in cand[:top_k]:
            u = it["unit"]
            out.append({
                "faiss_id": it["faiss_id"],
                "distance": it["distance"],
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
