# api_server.py (경로 문제 해결 및 최종 수정 완료)

import os
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import uvicorn
import json
import pathlib  # <--- pathlib 라이브러리 추가

# --- 서비스 모듈 임포트 ---
import gemini_service
import vector_search_service
import google.generativeai as genai

# --- 경로 및 초기 설정 ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
app = FastAPI()

# --- 서비스 초기화 ---
@app.on_event("startup")
def startup_event():
    logging.info("===== 서버 시작: 서비스 초기화 시작 =====")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("환경변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")
        genai.configure(api_key=api_key)
        logging.info("Gemini 서비스가 초기화되었습니다.")
        vector_search_service.load_index()
        logging.info("벡터 DB가 메모리에 로드되었습니다.")
        logging.info("===== 서버 시작: 서비스 초기화 완료 =====")
    except Exception as e:
        logging.critical(f"서버 시작 중 심각한 오류 발생: {e}", exc_info=True)
    logging.info("===== 서버 시작: 서비스 초기화 완료 =====")

# --- API 엔드포인트 ---
@app.get("/api/get-answer")
async def get_answer(q: str = Query(..., min_length=1, description="사용자 질문")):
    logging.info(f"===== 파이프라인 시작: 원본 질문 = \"{q}\" =====")
    try:
        transformed_query = gemini_service.transform_query(q)
        candidate_docs = vector_search_service.search_vectors(transformed_query, k=5)
        if not candidate_docs:
            logging.warning("벡터 DB에서 관련 문서를 찾지 못했습니다.")
            return {"summary": "관련 법령 정보를 찾을 수 없습니다.", "checklist": [], "source_document": "N/A"}
        logging.info(f"{len(candidate_docs)}개의 후보 문서를 검색했습니다.")
        for doc in candidate_docs:
            print(doc)
        final_answer = gemini_service.generate_final_answer(q, candidate_docs)
        logging.info("===== 파이프라인 종료 =====")
        return final_answer
    except Exception as e:
        logging.error(f"[api] - 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="답변 생성 중 서버 내부 오류가 발생했습니다.")
    
# --- 정적 파일 서빙 ---
# 상대 경로 대신, 위에서 정의한 절대 경로를 사용합니다. (수정된 부분)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    # 이제 이 파일을 직접 실행해도 화면과 API 모두 정상 작동합니다.
    uvicorn.run(app, host="127.0.0.1", port=8000)