# vector_search_service.py (수정된 전체 코드)

import json
import os
import logging
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any

# --- 상수 ---
KNOWLEDGE_BASE_FILE = "answers.json"
EMBEDDING_MODEL = "models/text-embedding-004"
FAISS_INDEX_FILE = "faiss_index.bin"
ID_MAP_FILE = "faiss_id_map.json"

# --- 전역 변수: 서버 시작 시 여기에 인덱스와 데이터를 로드합니다 ---
index = None
id_map = None
knowledge_base = None

def load_index():
    """
    서버 시작 시 호출될 함수.
    FAISS 인덱스와 원본 데이터를 메모리에 로드합니다.
    """
    global index, id_map, knowledge_base
    
    if not os.path.exists(FAISS_INDEX_FILE):
        logging.error(f"{FAISS_INDEX_FILE} 파일이 없습니다. build_db.py를 먼저 실행하세요.")
        return

    logging.info(f"'{FAISS_INDEX_FILE}'에서 인덱스를 로드합니다.")
    index = faiss.read_index(FAISS_INDEX_FILE)
    
    with open(ID_MAP_FILE, 'r', encoding='utf-8') as f:
        id_map = json.load(f)
    
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> List[float]:
    """Gemini API를 호출하여 텍스트 임베딩을 반환합니다."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type)
    return result['embedding']

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
        original_id = id_map.get(str(faiss_id))
        
        if original_id:
            results.append({
                "id": original_id,
                "score": distances[0][i], # NumPy float32 타입
                "data": knowledge_base[original_id]
            })
            
    return results

# build_and_save_index 함수는 수정할 필요가 없으므로 여기에 포함하지 않았습니다.
# 파일에 해당 함수가 있다면 그대로 두시면 됩니다.