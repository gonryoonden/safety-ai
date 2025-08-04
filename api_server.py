# api_server.py (최종 수정본)

from fastapi import FastAPI, Query, HTTPException
import logging
from dotenv import load_dotenv # 👈 1. 라이브러리 임포트

# 👈 2. FastAPI 앱을 생성하기 전에 가장 먼저 .env 파일을 로드합니다.
#    이렇게 하면 이 파일뿐만 아니라, 여기서 임포트하는 모든 다른 파일(vector_search_service 등)에서도
#    API 키를 정상적으로 사용할 수 있습니다.
load_dotenv()

import vector_search_service
import utils

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="산업안전 AI 어시스턴트 API",
    version="1.2.0",
    description="산업안전 지식에 대해 질문하고, 관련 법령 정보까지 얻을 수 있는 API입니다."
)

@app.on_event("startup")
def startup_event():
    """서버가 시작될 때 벡터 DB를 메모리에 로드하여 응답 속도를 높입니다."""
    try:
        vector_search_service.load_index()
        logging.info("벡터 인덱스를 성공적으로 로드했습니다.")
    except Exception as e:
        logging.error(f"서버 시작 중 벡터 인덱스 로딩 실패: {e}")
        # 여기서 서버를 중단시키거나, 에러 상태임을 알리는 로직을 추가할 수 있습니다.

@app.get("/search", summary="산업안전 지식 검색")
def search_knowledge_base(q: str = Query(..., description="검색할 질문 내용")):
    """
    사용자의 질문(q)을 받아 벡터 검색을 수행하고,
    가장 유사한 답변과 관련 법령 원문 링크를 반환합니다.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=422, detail="검색어(q)를 입력해주세요.")

    try:
        results = vector_search_service.search_vectors(q, k=1)

        if not results:
            return {"message": "관련 답변을 찾지 못했습니다."}

        top_result = results[0]
        
        # [해결] 유사도 점수를 파이썬 기본 float으로 변환합니다.
        top_result["score"] = float(top_result["score"])
        
        related_law_text = top_result.get("data", {}).get("연관조문", [""])[0]
        source_link = utils.make_law_link(related_law_text) if related_law_text else "연관 조문 없음"
        top_result["source_link"] = source_link

        return top_result

    except Exception as e:
        logging.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다.")