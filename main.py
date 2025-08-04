import requests
import json
import os
import re
import time
import logging
from typing import List, Dict, Optional, Any, Set

# [TF-IDF 도입] 필요한 라이브러리 임포트
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. 설정 및 상수 정의 ---

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 파일 및 API 관련 상수
KNOWLEDGE_BASE_FILE = "answers.json"
OC_VALUE = 'shg3335'  # 유효한 API 키
SEARCH_URL = 'http://www.law.go.kr/DRF/lawSearch.do'
CONTENT_URL = 'http://www.law.go.kr/DRF/lawService.do'

# 핵심 법령 세트
CORE_LAW_SET = [
    "산업안전보건법", "중대재해 처벌 등에 관한 법률", "산업안전보건법 시행령",
    "산업안전보건법 시행규칙", "산업안전보건기준에 관한 규칙", "중대재해 처벌 등에 관한 법률 시행령",
    "위험물안전관리법", "액화석유가스의 안전관리 및 사업법", "도시가스사업법",
    "건설기술 진흥법", "건축법", "건설기계관리법", "산업재해보상보험법"
]

# --- [핵심 수정] TfidfVectorizer가 사용할 불용어 리스트를 다시 정의 ---
STOPWORDS: Set[str] = {
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '에게', '께', '한테',
    '으로', '로', '와', '과', '고', '하다', '이다', '되다'
}
# --- 수정 완료 ---

# --- [핵심 수정] TfidfVectorizer 생성 시 불용어 리스트를 전달 ---
vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))
# --- 수정 완료 ---

tfidf_matrix = None
kb_data_list = []

# --- 2. 로컬 지식 베이스(JSON) 처리 함수 ---

def load_and_prepare_knowledge_base(file_path: str):
    """[최종 수정] 지식 베이스를 로드하고, '키워드' 필드를 포함하여 TF-IDF 모델을 학습시킵니다."""
    global tfidf_matrix, kb_data_list
    if not os.path.exists(file_path):
        logging.error(f"'{file_path}' 파일을 찾을 수 없습니다.")
        raise FileNotFoundError(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
    try:
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
        return kb_dict
    except json.JSONDecodeError:
        logging.error(f"'{file_path}' 파일의 JSON 형식이 올바지 않습니다.")
        raise ValueError(f"오류: '{file_path}' 파일의 JSON 형식이 올바르지 않습니다.")


def search_knowledge_base_tfidf(query: str) -> List[Dict[str, Any]]:
    """[다중 답변 반환] TF-IDF와 코사인 유사도를 사용하여 가장 적합한 문서들을 찾습니다."""
    global vectorizer, tfidf_matrix, kb_data_list
    results = []
    if not query.strip():
        return results
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    SIMILARITY_THRESHOLD = 0.1
    for i in related_docs_indices:
        if cosine_similarities[i] > SIMILARITY_THRESHOLD and len(results) < 3:
            match_item = kb_data_list[i].copy() # 원본 수정을 방지하기 위해 복사본 사용
            match_item['similarity_score'] = cosine_similarities[i]
            results.append(match_item)
    if results:
        logging.info(f"검색 완료. 총 {len(results)}개의 유사 항목을 찾았습니다.")
    else:
        logging.warning(f"검색 결과, 일치하는 항목을 찾지 못했습니다.")
    return results

def format_answer(item: Dict[str, Any], rank: int) -> str:
    """찾아낸 지식 베이스 항목을 지정된 형식의 문자열로 가공합니다."""
    summary = "\n".join(f"- {line}" for line in item.get("적용상황", ["요약 정보가 없습니다."]))
    practical_point = item.get("실무포인트", "실무 포인트 정보가 없습니다.")
    score = item.get('similarity_score', 0)
    legal_basis_obj = item.get("기본조문", {})
    if isinstance(legal_basis_obj, dict):
        law_name = legal_basis_obj.get('법령명', 'N/A')
        article = legal_basis_obj.get('조문', 'N/A')
        legal_basis_str = f"**{law_name} {article}**"
    else:
        legal_basis_str = "관련 법령 정보가 없습니다."
    formatted_output = f"""
### Kandidat {rank} (유사도: {score:.2f}) - {item.get('핵심질문', 'N/A')}
---
#### 💡 적용 상황
{summary}
#### ⚖️ 법령 근거
{legal_basis_str}
#### ✅ 현장 실무 포인트
{practical_point}
"""
    return formatted_output.strip()

# --- 3. 법령 API 조회(lawFetcher) 관련 함수 (이전과 동일) ---
def search_law_api(law_name: str) -> Optional[Dict[str, Any]]:
    # ... (이전 코드와 동일)
    pass 

# --- 4. 메인 실행 로직 ---
def run_local_search():
    """로컬 지식 베이스를 검색하는 모드를 실행합니다."""
    question = input("질문을 입력하세요: ")
    logging.info(f"사용자 질문: \"{question}\"")
    found_items = search_knowledge_base_tfidf(question)
    print("\n" + "="*40)
    if found_items:
        print(f"✅ AI 어시스턴트 답변입니다. (총 {len(found_items)}개)")
        for i, item in enumerate(found_items):
            final_answer = format_answer(item, i + 1)
            print(final_answer)
            print("\n" + "-"*40)
        print("📚 출처: 내부 전문가 지식 베이스 (answers.json)")
    else:
        print("⚠️ 죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다.")
    print("="*40)

def run_api_fetcher():
    # ... (이전 코드와 동일)
    pass

def main():
    """메인 실행 함수"""
    try:
        load_and_prepare_knowledge_base(KNOWLEDGE_BASE_FILE)
        while True:
            print("\n어떤 작업을 수행하시겠습니까?")
            mode = input("1: 로컬 질문 검색, 2: 법령 API 데이터 구축, 3: 종료 (숫자 입력): ")
            if mode == '1':
                run_local_search()
            elif mode == '2':
                run_api_fetcher()
            elif mode == '3':
                break
            else:
                print("잘못된 입력입니다. 1, 2, 3 중 하나를 입력하세요.")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[실행 오류] 프로그램을 시작할 수 없습니다. ({e})")
    except (KeyboardInterrupt, EOFError):
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()