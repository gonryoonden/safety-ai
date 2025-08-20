import os
import requests
import json
import re
import time

# --- 설정 ---
# 📌 크롤링할 법령 목록 (요청하신 목록으로 업데이트)
LAW_NAMES_TO_CRAWL = [
    "산업안전보건법",
    "산업안전보건법 시행령",
    "산업안전보건법 시행규칙",
    "산업안전보건기준에 관한 규칙",
    "중대재해 처벌 등에 관한 법률",
    "중대재해 처벌 등에 관한 법률 시행령",
    "산업재해보상보험법"
]

# 결과를 저장할 디렉토리 이름
OUTPUT_DIR = "crawled_laws"

# API 요청에 필요한 OC 값
LAW_API_OC = os.getenv("LAW_API_OC", "test")

# --- 함수 ---
def sanitize_filename(filename: str) -> str:
    """파일명으로 사용할 수 없는 문자를 제거합니다."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def crawl_and_save_law(law_name: str):
    """지정된 법령을 API로 조회하고 별도 파일로 저장합니다."""
    
    api_url = (
        "http://www.law.go.kr/DRF/lawSearch.do"
        f"?OC={LAW_API_OC}"
        "&target=law"
        "&type=JSON"
        f"&query={law_name}"
    )

    print(f"'{law_name}' 법령 정보 수집 중...")

    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 처리
        law_data = response.json()

        # 법령 검색 결과가 있는지 확인
        if not law_data.get("LawSearch", {}).get("law"):
            print(f"경고: '{law_name}'에 대한 검색 결과가 없습니다. 법령명을 다시 확인해주세요.")
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for law in LAW_NAMES_TO_CRAWL:
        crawl_and_save_law(law)
        time.sleep(0.5) # API 서버 부하 감소를 위해 요청 사이에 간단한 지연 추가
    
    print("\n모든 법령 정보 수집을 완료했습니다.")