# search_service.py

import json
import os
import logging
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 설정
KNOWLEDGE_BASE_FILE = "answers.json"
STOPWORDS: set = {
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '에게', '께', '한테',
    '으로', '로', '와', '과', '고', '하다', '이다', '되다'
}

# 전역 변수
vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
tfidf_matrix = None
kb_data_list = []

def load_and_prepare_knowledge_base():
    """지식 베이스를 로드하고, TF-IDF 모델을 학습시킵니다."""
    global tfidf_matrix, kb_data_list
    
    file_path = KNOWLEDGE_BASE_FILE
    if not os.path.exists(file_path):
        logging.error(f"'{file_path}' 파일을 찾을 수 없습니다.")
        raise FileNotFoundError(f"'{file_path}' 파일을 찾을 수 없습니다.")

    with open(file_path, 'r', encoding='utf-8') as f:
        kb_dict = json.load(f)
    
    kb_data_list = list(kb_dict.values())
    documents = []
    for item in kb_data_list:
        question_text = item.get("핵심질문", "")
        summary_text = " ".join(item.get("적용상황", []))
        point_text = item.get("실무포인트", "")
        keywords = " ".join(item.get("키워드", []))
        search_corpus = f"{question_text} {summary_text} {point_text} {keywords} {keywords}"
        documents.append(search_corpus)

    tfidf_matrix = vectorizer.fit_transform(documents)
    logging.info(f"지식 베이스 로딩 및 TF-IDF 모델 준비 완료. 총 {len(kb_data_list)}개 항목.")

def search_documents(query: str) -> List[Dict[str, Any]]:
    """사용자 질문으로 가장 유사한 문서들을 검색하여 반환합니다."""
    results = []
    if not query.strip():
        return results

    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]

    SIMILARITY_THRESHOLD = 0.1
    for i in related_docs_indices:
        if cosine_similarities[i] > SIMILARITY_THRESHOLD and len(results) < 3:
            match_item = kb_data_list[i].copy()
            match_item['similarity_score'] = cosine_similarities[i]
            results.append(match_item)
            
    return results
    