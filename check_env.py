import os
from dotenv import load_dotenv

# .env 파일을 로드합니다.
load_dotenv()

# 'GEMINI_API_KEY' 라는 이름의 환경 변수를 가져옵니다.
api_key = os.getenv("GEMINI_API_KEY")

# 결과를 출력합니다.
if api_key:
    # 키의 일부만 출력하여 유출을 방지합니다.
    print(f"✅ 성공: API 키를 찾았습니다. (시작 4자리: {api_key[:4]}...)")
else:
    print("❌ 실패: .env 파일에서 'GEMINI_API_KEY'를 찾지 못했습니다.")
    print("'.env' 파일의 내용이 'GEMINI_API_KEY=\"...\"' 형식인지 다시 확인해 주세요.")