# vector_search_service.py (수정된 버전)

import json
import os
import logging
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any

# --- 상수 변경 ---
KNOWLEDGE_BASE_FILE = "answers.json"
# 로컬 모델 이름 대신 Gemini 모델 사용
EMBEDDING_MODEL = "models/text-embedding-004" 
FAISS_INDEX_FILE = "faiss_index.bin"
INDEX_DIMENSION = 768 # text-embedding-004 모델의 임베딩 차원

# --- API 키 설정 ---
# build_db.py 또는 api_server.py에서 API 키를 로드하므로 여기서 직접 로드할 필요는 없습니다.
# 다만, genai.configure()는 API를 사용하기 전에 반드시 호출되어야 합니다.
# 이 서비스가 로드될 때 설정되도록 구성합니다.
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logging.warning("GEMINI_API_KEY 환경변수가 설정되지 않았습니다. API 호출 시 오류가 발생할 수 있습니다.")


def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> List[float]:
    """Gemini API를 호출하여 텍스트 임베딩을 반환합니다."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type)
    return result['embedding']

def build_and_save_index():
    """answers.json을 읽어 Gemini 임베딩으로 Faiss 인덱스를 빌드하고 저장합니다."""
    logging.info(f"'{KNOWLEDGE_BASE_FILE}'에서 데이터를 로드합니다.")
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 텍스트와 ID를 추출합니다.
    # '핵심질문' + '실무포인트'를 합쳐서 임베딩의 의미를 풍부하게 합니다.
    ids = list(data.keys())
    texts_to_embed = [item['핵심질문'] + " " + item['실무포인트'] for item in data.values()]
    
    logging.info(f"총 {len(texts_to_embed)}개의 텍스트에 대해 Gemini 임베딩을 생성합니다.")
    
    # Gemini API 호출
    embeddings = [get_embedding(text) for text in texts_to_embed]
    
    # Faiss 인덱스 생성
    index = faiss.IndexFlatL2(INDEX_DIMENSION)
    index_id_map = faiss.IndexIDMap(index)
    
    # 벡터와 ID를 인덱스에 추가
    np_embeddings = np.array(embeddings, dtype='float32')
    np_ids = np.array(list(range(len(ids)))).astype('int64') # Faiss는 숫자 ID를 사용
    
    index_id_map.add_with_ids(np_embeddings, np_ids)
    
    logging.info(f"'{FAISS_INDEX_FILE}' 파일로 인덱스를 저장합니다.")
    faiss.write_index(index_id_map, FAISS_INDEX_FILE)
    
    # ID와 Faiss 내부 ID 매핑 저장 (검색 결과 복원을 위함)
    with open('faiss_id_map.json', 'w', encoding='utf-8') as f:
        id_mapping = {i: original_id for i, original_id in enumerate(ids)}
        json.dump(id_mapping, f)


def search_vectors(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """사용자 쿼리를 임베딩하고 Faiss에서 가장 유사한 k개의 결과를 찾습니다."""
    if not os.path.exists(FAISS_INDEX_FILE):
        logging.error("Faiss 인덱스 파일이 없습니다. 먼저 DB를 빌드하세요.")
        return []

    # 인덱스와 ID 맵 로드
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open('faiss_id_map.json', 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    # 쿼리 임베딩 (검색 쿼리는 다른 task_type 사용)
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    query_vector = np.array([query_embedding], dtype='float32')
    
    # 검색 수행
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i in range(len(indices[0])):
        faiss_id = indices[0][i]
        original_id = id_mapping.get(str(faiss_id)) # JSON 키는 문자열
        if original_id:
            results.append({
                "id": original_id,
                "score": distances[0][i],
                "data": knowledge_base[original_id]
            })
            
    return results