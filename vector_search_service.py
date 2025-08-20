import json
import os
import logging
import faiss
import numpy as np
import google.generativeai as genai
import requests
import time
import re
from typing import List, Dict, Any, Tuple

# --- 상수 ---
RAW_DATA_FILE = "law_database.json"
KNOWLEDGE_BASE_FILE = "answers.json"
EMBEDDING_MODEL = "models/text-embedding-004"
FAISS_INDEX_FILE = "faiss_index.bin"
ID_MAP_FILE = "faiss_id_map.json"
LAW_API_OC = os.getenv("LAW_API_OC", "test")

# --- 전역 변수 ---
index = None
id_map = None
knowledge_base = None

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# ==================================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 영역 START ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==================================================================================

def format_jo_no(jo_no_str: str) -> str:
    """조문 번호를 API가 요구하는 6자리 형식으로 변환합니다. (예: '2' -> '000200', '4의2' -> '000402')"""
    parts = jo_no_str.split('의')
    main_no = parts[0].zfill(4)
    sub_no = parts[1].zfill(2) if len(parts) > 1 else "00"
    return main_no + sub_no

def flatten_api_response(article_detail: Dict[str, Any]) -> str:
    """
    법령 본문 API의 계층적 JSON 데이터를 재귀적으로 파싱하여,
    조(條), 항(項), 호(號), 목(目)까지 모든 텍스트를 구조적으로 추출합니다.
    (들여쓰기 강화 최종 버전)
    """

    def ensure_list(obj):
        if obj is None: return []
        if isinstance(obj, list): return obj
        return [obj]

    def parse_mok(mok_list):
        result = []
        for mok in ensure_list(mok_list):
            mok_content = mok.get('목내용', '').strip()
            if mok_content:
                # 목(目)은 4칸 들여쓰기
                result.append(f"    {mok_content}")
        return result

    def parse_ho(ho_list):
        result = []
        for ho in ensure_list(ho_list):
            ho_content = ho.get('호내용', '').strip()
            line_parts = []
            if ho_content:
                # 호(號)는 2칸 들여쓰기
                line_parts.append(f"  {ho_content}")

            mok_items = ho.get('목')
            if mok_items:
                mok_lines = parse_mok(mok_items)
                if mok_lines:
                    line_parts.append('\n'.join(mok_lines))

            if line_parts:
                result.append('\n'.join(line_parts))
        return result

    def parse_hang(hang_list):
        result = []
        for hang in ensure_list(hang_list):
            hang_content = hang.get('항내용', '').strip()
            line_parts = []
            if hang_content:
                line_parts.append(hang_content)

            ho_items = hang.get('호')
            if ho_items:
                ho_lines = parse_ho(ho_items)
                if ho_lines:
                    line_parts.append('\n'.join(ho_lines))

            if line_parts:
                result.append('\n'.join(line_parts))
        return result

    # --- 조문(條) 계층 처리 ---
    jo_no = article_detail.get('조문번호', '')
    jo_title = article_detail.get('조문제목', '')
    jo_content = article_detail.get('조문내용', '').strip()

    result_lines = []

    if jo_content:
        result_lines.append(jo_content)
    else:
        jo_prefix = f"제{jo_no}조"
        if jo_title:
            jo_prefix += f"({jo_title})"
        result_lines.append(jo_prefix)

    hang_items = article_detail.get('항')
    if hang_items:
        hang_lines = parse_hang(hang_items)
        if hang_lines:
            result_lines.append('\n'.join(hang_lines))

    return '\n'.join(result_lines)

def fetch_law_details(law_mst: str) -> List[Dict[str, Any]]:
    """2단계 API 호출을 통해 법령의 상세하고 구조화된 조문 데이터를 가져옵니다."""
    # 1단계: 법령의 전체 조문 목록 확보
    list_url = f"http://www.law.go.kr/DRF/lawService.do?OC={LAW_API_OC}&target=law&MST={law_mst}&type=JSON"
    logging.info(f"Fetching list_url: {list_url}")
    try:
        list_response = requests.get(list_url, timeout=30)
        logging.info(f"List response status: {list_response.status_code}")
        logging.info(f"List response content (first 500 chars): {list_response.text[:500]}")
        list_response.raise_for_status()
        list_data = list_response.json()
        articles_summary = list_data.get("법령", {}).get("조문", {}).get("조문단위", [])
        if not articles_summary:
            logging.warning(f"조문 목록을 찾을 수 없습니다. (MST: {law_mst})")
            return []
    except Exception as e:
        logging.error(f"조문 목록 조회 실패 (MST: {law_mst}): {e}")
        return []

    all_parsed_articles = []
    # 2단계: 각 조문에 대해 상세 정보 요청
    for article_summary in articles_summary:
        jo_no_from_list = article_summary.get("조문번호")
        if not jo_no_from_list: continue

        formatted_jo = format_jo_no(jo_no_from_list)
        detail_url = f"http://www.law.go.kr/DRF/lawService.do?OC={LAW_API_OC}&target=law&MST={law_mst}&JO={formatted_jo}&type=JSON"
        logging.info(f"Fetching detail_url: {detail_url}")
        try:
            time.sleep(0.1) # API 서버 부하 감소
            detail_response = requests.get(detail_url, timeout=30)
            logging.info(f"Detail response status: {detail_response.status_code}")
            logging.info(f"Detail response content (first 500 chars): {detail_response.text[:500]}")
            detail_response.raise_for_status()
            detail_data = detail_response.json()

            # 상세 조회 API는 보통 단일 "법령" 객체를 반환
            law_detail = detail_data.get("법령", {})
            if not law_detail: continue

            # 조문 정보 추출 (리스트일 수 있으므로 첫번째 요소 사용)
            jo_details = law_detail.get("조문", {}).get("조문단위", [])
            if not jo_details: continue
            
            first_jo_detail = {}
            if isinstance(jo_details, list) and jo_details:
                first_jo_detail = jo_details[0]
            elif isinstance(jo_details, dict):
                first_jo_detail = jo_details

            # 임베딩용 평탄화 텍스트 생성
            embedding_text = flatten_api_response(first_jo_detail)

            # Extract actual article number from embedding_text
            # The regex is 제(\d+조(?:의\d+)?)
            match = re.search(r"제(\d+조(?:의\d+)?)", embedding_text)
            extracted_jo_no = match.group(1) if match else first_jo_detail.get("조문번호") # Fallback to API's jo_no if not found

            all_parsed_articles.append({
                "조문번호": extracted_jo_no, # Use the extracted jo_no
                "조문제목": first_jo_detail.get("조문제목"),
                "조문내용": first_jo_detail, # API가 직접 구조화한 전체 데이터
                "text_for_embedding": embedding_text
            })

        except Exception as e:
            logging.error(f"조문 상세 조회 실패 (MST: {law_mst}, 조문: {jo_no_from_list}): {e}")
            continue

    logging.info(f"성공적으로 법령 본문을 파싱했습니다. (MST: {law_mst}, 조문 수: {len(all_parsed_articles)})")
    return all_parsed_articles

def build_and_save_index(limit: int = None):
    logging.info("'laws' 디렉터리에서 법령별 개별 DB 생성을 시작합니다.")

    law_files = [os.path.join("laws", f) for f in os.listdir("laws") if f.endswith(".json")]
    total_files = len(law_files)
    processed_files_count = 0

    for law_file_path in law_files:
        if limit is not None and processed_files_count >= limit:
            logging.info(f"Limit of {limit} files reached. Stopping processing.")
            break

        processed_files_count += 1
        base_filename = os.path.splitext(os.path.basename(law_file_path))[0]
        logging.info(f"({processed_files_count}/{total_files}) '{base_filename}.json' 파일 처리 시작...")

        current_knowledge_base = {}
        current_texts_to_embed_map = {}

        try:
            with open(law_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"'{law_file_path}' 파일 로드 또는 파싱 실패: {e}")
            continue

        law_items = raw_data.get("LawSearch", {}).get("law", [])
        if not isinstance(law_items, list):
            law_items = [law_items]

        if not law_items: continue
        item = law_items[0]

        law_mst = item.get("법령일련번호")
        law_id = item.get("법령ID") # 법령상세링크 생성을 위해 법령ID 사용
        if not law_mst or not law_id: continue

        article_chunks = fetch_law_details(law_mst)

        for chunk in article_chunks:
            unique_id = f"{law_mst}-{chunk['조문번호']}"
            current_knowledge_base[unique_id] = {
                "법령명한글": item.get("법령명한글", ""),
                "법령상세링크": f"http://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={law_id}",
                "조문번호": chunk['조문번호'],
                "조문제목": chunk['조문제목'],
                "조문내용": chunk['text_for_embedding']
            }
            current_texts_to_embed_map[unique_id] = chunk['text_for_embedding']

        if not current_knowledge_base:
            logging.warning(f"'{base_filename}.json'에서 처리할 조문 데이터가 없습니다. 건너뜁니다.")
            continue

        output_kb_file = f"{base_filename}_answers.json"
        output_index_file = f"{base_filename}_faiss_index.bin"
        output_id_map_file = f"{base_filename}_faiss_id_map.json"

        logging.info(f"총 {len(current_knowledge_base)}개 조문 수집 완료. '{output_kb_file}' 파일로 저장합니다.")
        with open(output_kb_file, 'w', encoding='utf-8') as f:
            json.dump(current_knowledge_base, f, ensure_ascii=False, indent=4)

        logging.info(f"'{base_filename}'에 대한 Faiss 인덱스 생성을 시작합니다...")
        texts_to_embed = list(current_texts_to_embed_map.values())
        if not texts_to_embed: continue

        faiss_ids = list(range(len(texts_to_embed)))
        temp_id_map = {i: k for i, k in enumerate(current_texts_to_embed_map.keys())}

        embeddings = genai.embed_content(model=EMBEDDING_MODEL, content=texts_to_embed, task_type="RETRIEVAL_DOCUMENT")['embedding']
        embeddings_np = np.array(embeddings, dtype='float32')
        d = embeddings_np.shape[1]
        new_index = faiss.IndexFlatL2(d)
        new_index = faiss.IndexIDMap(new_index)
        new_index.add_with_ids(embeddings_np, np.array(faiss_ids))

        faiss.write_index(new_index, output_index_file)
        logging.info(f"Faiss 인덱스를 '{output_index_file}'에 저장했습니다.")

        with open(output_id_map_file, 'w', encoding='utf-8') as f:
            json.dump(temp_id_map, f)
        logging.info(f"ID 맵을 '{output_id_map_file}'에 저장했습니다.")

    logging.info("모든 법령 파일에 대한 개별 DB 생성이 완료되었습니다.")

# ==================================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 핵심 수정 영역 END ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==================================================================================

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    """Gemini API를 호출하여 텍스트 임베딩을 반환합니다."""
    if task_type == "RETRIEVAL_DOCUMENT":
        title = "산업안전보건법 관련 조항" if "산업안전" in text else "법률 조항"
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            title=title,
            task_type=task_type)
    else:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type)
    return result['embedding']

def load_index():
    """
    서버 시작 시 호출될 함수.
    FAISS 인덱스와 원본 데이터를 메모리에 로드합니다.
    """
    global index, id_map, knowledge_base
    required_files = [FAISS_INDEX_FILE, ID_MAP_FILE, KNOWLEDGE_BASE_FILE]
    for f in required_files:
        if not os.path.exists(f):
            logging.error(f"'{f}' 파일이 없습니다. build_db.py를 먼저 실행하여 DB를 생성하세요.")
            return

    logging.info(f"'{FAISS_INDEX_FILE}'에서 인덱스를 로드합니다.")
    index = faiss.read_index(FAISS_INDEX_FILE)

    logging.info(f"'{ID_MAP_FILE}'에서 ID맵을 로드합니다.")
    with open(ID_MAP_FILE, 'r', encoding='utf-8') as f:
        id_map = {int(k): v for k, v in json.load(f).items()}

    logging.info(f"'{KNOWLEDGE_BASE_FILE}'에서 knowledge base를 로드합니다.")
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

def search_vectors(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """사용자 쿼리를 임베딩하고 미리 로드된 Faiss 인덱스에서 가장 유사한 k개의 결과를 찾습니다."""
    if index is None:
        logging.error("인덱스가 로드되지 않았습니다. 서버 로그를 확인하세요.")
        return []

    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    query_vector = np.array([query_embedding], dtype='float32')

    distances, indices = index.search(query_vector, k)

    results = []
    for i in range(len(indices[0])):
        if indices[0][i] == -1:
            continue

        faiss_id = indices[0][i]
        original_id = id_map.get(faiss_id)

        if original_id and original_id in knowledge_base:
            retrieved_data = knowledge_base[original_id]
            results.append({
                "id": original_id,
                "score": float(distances[0][i]),
                "data": retrieved_data
            })

    return results
