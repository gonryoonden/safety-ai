import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import json

# --- 서비스 모듈 임포트 ---
# 각 파일이 동일한 디렉토리에 있다고 가정
import gemini_service
import vector_search_service

# --- 초기 설정 ---
# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def initialize_services():
    """
    스크립트 실행 시 필요한 서비스들을 초기화합니다.
    """
    # Gemini API 키 설정
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    genai.configure(api_key=api_key)
    logging.info("Gemini 서비스가 초기화되었습니다.")

    # 벡터 DB 로드
    vector_search_service.load_index()
    logging.info("벡터 DB가 메모리에 로드되었습니다.")


def run_pipeline(user_query: str):
    """
    사용자 질문을 입력받아 전체 RAG 파이프라인을 실행합니다.
    """
    logging.info(f"===== 파이프라인 시작: 원본 질문 = \"{user_query}\" =====")

    # 1단계: 질의 변환
    transformed_query = gemini_service.transform_query(user_query)
    
    # 2단계: 후보군 검색
    logging.info("벡터 DB에서 후보군 검색 시작...")
    # 질의가 변환되었으므로, 더 많은 후보군(k=5)을 검색하여 2차 Gemini에게 풍부한 컨텍스트 제공
    candidate_docs = vector_search_service.search_vectors(transformed_query, k=5)
    if not candidate_docs:
        logging.warning("벡터 DB에서 관련 문서를 찾지 못했습니다.")
        return {"summary": "관련 법령 정보를 찾을 수 없습니다.", "checklist": [], "source_document": ""}
    logging.info(f"{len(candidate_docs)}개의 후보 문서를 검색했습니다.")

    # 3단계: 최종 답변 생성
    final_answer = gemini_service.generate_final_answer(user_query, candidate_docs)

    logging.info("===== 파이프라인 종료 =====")
    return final_answer


if __name__ == "__main__":
    # 1. 서비스 초기화
    initialize_services()

    # 2. 테스트 질문으로 파이프라인 실행
    test_query = "타워크레인 설치할 때 뭘 지켜야 하고 어떤 서류가 필요한가요?"
    result = run_pipeline(test_query)

    # 3. 최종 결과 출력
    print("\n[최종 생성된 답변]")
    print(json.dumps(result, indent=2, ensure_ascii=False))