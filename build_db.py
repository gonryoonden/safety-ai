# build_db.py (수정 완료된 최종 코드)

import logging
import os
from dotenv import load_dotenv

# 1. .env 파일을 먼저 로드합니다.
load_dotenv()

# 2. 그 다음에 .env 변수를 사용하는 모듈을 임포트합니다.
import vector_search_service

# 로그 설정
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        logging.error("환경변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    else:
        logging.info("벡터 DB 생성을 시작합니다...")
        try:
            vector_search_service.build_and_save_index()
            logging.info("✅ 성공적으로 'faiss_index.bin' 파일을 생성했습니다.")
        except Exception as e:
            logging.error(f"벡터 DB 생성 중 오류 발생: {e}")