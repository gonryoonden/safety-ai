import os
import json
import requests
import re
import time
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 설정 ---
# 1단계 크롤링 결과가 저장된 디렉토리
INPUT_DIR = "crawled_laws"
# 2단계 상세 본문 크롤링 결과를 저장할 디렉토리
OUTPUT_DIR = "crawled_laws_details"

# API 요청에 필요한 OC 값 (환경 변수에서 가져오기)
LAW_API_OC = os.getenv("LAW_API_OC", "test")

# --- 함수 ---
def sanitize_filename(filename: str) -> str:
    """파일명으로 사용할 수 없는 문자를 제거합니다."""
    return re.sub(r'[\\/*?:\"<>|]', "", filename)

def crawl_and_save_details(law_id: str, law_name: str):
    """지정된 법령 ID를 사용하여 상세 본문을 API로 조회하고 별도 파일로 저장합니다."""
    
    api_url = (
        "http://www.law.go.kr/DRF/lawService.do"
        f"?OC={LAW_API_OC}"
        "&target=law"
        "&type=JSON"
        f"&ID={law_id}"
    )

    print(f"'{law_name}' (ID:{law_id}) 상세 정보 수집 중...")

    try:
        response = requests.get(api_url, timeout=15) # 타임아웃을 15초로 단축하여 빠른 피드백 확인
        response.raise_for_status()
        law_data = response.json()

        # 유효한 데이터가 있는지 확인
        if not law_data.get("기본정보"):
            print(f"경고: '{law_name}' (ID:{law_id})에 대한 상세 정보가 없습니다. API 응답을 확인합니다.")
            print("--- API 응답 시작 ---")
            print(json.dumps(law_data, indent=4, ensure_ascii=False))
            print("--- API 응답 종료 ---")
            return

        safe_filename = sanitize_filename(law_name) + ".json"
        output_path = os.path.join(OUTPUT_DIR, safe_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(law_data, f, ensure_ascii=False, indent=4) 
        
        print(f"성공: '{output_path}'에 저장되었습니다.")

    except requests.exceptions.RequestException as e:
        print(f"실패: '{law_name}' 수집 중 네트워크 오류 발생: {e}")
    except json.JSONDecodeError:
        print(f"실패: '{law_name}' 수집 후 JSON 파싱 오류 발생 (응답이 비어있을 수 있습니다).")
    except Exception as e:
        print(f"실패: '{law_name}' 처리 중 알 수 없는 오류 발생: {e}")

# --- 메인 실행 ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"오류: 입력 디렉토리 '{INPUT_DIR}'를 찾을 수 없습니다. 1단계 크롤링을 먼저 실행해야 합니다.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        crawled_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
        
        print(f"총 {len(crawled_files)}개의 파일에서 법령 ID를 추출하여 상세 정보 수집을 시작합니다.")
        
        for filename in crawled_files:
            filepath = os.path.join(INPUT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # API 응답 구조에 따라 law_items 추출
            law_search_result = data.get("LawSearch", {})
            law_items = law_search_result.get("law", [])
            
            # law_items가 단일 dict일 경우 list로 감싸서 처리
            if isinstance(law_items, dict):
                law_items = [law_items]

            if not law_items:
                print(f"경고: '{filename}' 파일에 법령 정보가 없습니다.")
                continue

            for law in law_items:
                law_id = law.get("법령ID")
                law_name = law.get("법령명한글")
                
                if law_id and law_name:
                    crawl_and_save_details(law_id, law_name)
                    time.sleep(0.5) # API 서버 부하 감소
                else:
                    print(f"경고: '{filename}' 파일의 항목에서 법령ID 또는 법령명을 찾을 수 없습니다.")
        
        print("\n모든 법령 상세 정보 수집을 완료했습니다.")
