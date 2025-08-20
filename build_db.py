# build_db.py (수정 완료된 최종 코드)

import logging
import os
import re # Added for the manual loader

# --- Manual .env loader ---
def manual_load_dotenv(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'^([^=]+)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    if value.startswith(('"', "'")) and value.endswith(("'", "'")):
                        if value[0] == value[-1]:
                            value = value[1:-1]
                    os.environ[key] = value
    except FileNotFoundError:
        logging.error(f"Dotenv file not found at {path}")
    except Exception as e:
        logging.error(f"An error occurred during manual .env loading: {e}")

# 1. Manually load the .env file
dotenv_path = "C:\\Users\\kwater\\Desktop\\safety-ai\\.env"
manual_load_dotenv(dotenv_path)

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
            vector_search_service.build_and_save_index(limit=None)
            logging.info("성공적으로 'faiss_index.bin' 파일을 생성했습니다.") # Removed emoji
        except Exception as e:
            logging.error(f"벡터 DB 생성 중 오류 발생: {e}")
