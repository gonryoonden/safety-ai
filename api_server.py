# api_server.py

from fastapi import FastAPI, HTTPException, Query
import logging

# [수정] search_service 대신 vector_search_service를 임포트
import vector_search_service

# FastAPI 앱 생성
app = FastAPI(
    title="산업안전 AI 어시스턴트 API",
    version="v1.0.0", # 버전 업데이트
    description="[고도화] Sentence-BERT와 FAISS 기반의 시맨틱 검색 API"
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.on_event("startup")
def startup_event():
    """서버가 시작될 때 벡터 DB를 로드(또는 생성)합니다."""
    try:
        vector_search_service.load_index()
    except Exception as e:
        logging.error(f"서버 시작 중 벡터 인덱스 로딩 실패: {e}")

@app.get("/search")
def search_knowledge_base(q: str = Query(..., description="검색할 사용자 질문 (예: 작업발판 관련 법령 보여줘)")):
    """지식 베이스에서 사용자 질문(q)과 의미적으로 가장 유사한 답변을 검색합니다."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="필수 파라미터 'q'가 비어있습니다.")
    
    results = vector_search_service.search_vectors(q)
    
    if not results:
         return {"results": []}

    # 응답 데이터 구조 재구성
    response_data = []
    for i, item in enumerate(results):
        response_data.append({
            "rank": i + 1,
            "similarity_score": item.get('similarity_score', 0),
            "question": item.get('핵심질문', 'N/A'),
            "context": item.get('적용상황', []),
            "legal_basis": [item.get('기본조문')],
            "practical_point": item.get('실무포인트', 'N/A')
        })
        
    return {"results": response_data}