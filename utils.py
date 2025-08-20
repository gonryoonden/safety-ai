# utils.py (전문가 의견을 반영한 최종 수정본)
import re
import requests
import os
import logging
import json

def make_law_link(law_text: str) -> str:
    """
    법령 텍스트를 입력받아 국가법령정보센터 API를 실시간으로 조회하여
    가장 정확한 법령 원문 URL을 생성하여 반환합니다.
    """
    base_url = "http://www.law.go.kr"
    api_endpoint = f"{base_url}/DRF/lawSearch.do" # 👈 URL 기본 주소만 정의
    
    law_name = law_text.strip()
    
    oc_key = os.getenv("LAW_API_OC")
    if not oc_key:
        logging.error("LAW_API_OC 환경변수가 설정되지 않았습니다.")
        return base_url

    # 💡 [핵심 수정] 파라미터를 딕셔너리 객체로 분리
    params = {
        'OC': oc_key,
        'target': 'law',
        'type': 'JSON',
        'query': law_name
    }

    # 💡 [핵심 수정] User-Agent 헤더 추가
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    logging.info(f"API 요청 params: {params}")
    logging.info(f"API 요청 url: {api_endpoint}")

    try:
        # 💡 [핵심 수정] requests.get 호출 시 url, params, headers를 분리하여 전달
        response = requests.get(api_endpoint, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data and data.get("LawSearch", {}).get("law"):
            law_list = data["LawSearch"]["law"]
            # 💡 [수정] law가 단일 객체로 오는 경우도 처리
            if isinstance(law_list, dict):
                law_list = [law_list]
                
            if law_list:
                relative_link = law_list[0].get("법령상세링크")
                if relative_link:
                    return f"{base_url}{relative_link}"

    except json.JSONDecodeError:
        logging.error(f"법령 API가 JSON이 아닌 응답을 반환했습니다. [상태 코드: {response.status_code}]")
        logging.error(response.text)
    except requests.exceptions.RequestException as e:
        logging.error(f"법령 API 호출 중 네트워크 오류 발생: {e}")
    except (KeyError, IndexError) as e:
        logging.error(f"법령 API 응답 처리 중 데이터 구조 오류 발생: {e}")

    return f"{base_url}/LSW/lsSc.do?query={law_text}"
# utils.py 파일에 추가할 코드

import requests
import os

# LAW_API_OC는 .env 파일 등에서 안전하게 불러오는 것을 권장합니다.
# 여기서는 임시로 os.getenv를 사용합니다.
LAW_API_OC = os.getenv("LAW_API_OC", "test")

def get_attachment_link(law_name: str, attachment_number: str) -> str | None:
    """
    법령명과 별표 번호로 해당 별표/서식의 PDF 링크를 찾아 반환합니다.
    """
    print(f"'{law_name}'의 별표 '{attachment_number}'를 검색합니다...")
    
    # 1. API URL 구성
    url = "http://www.law.go.kr/DRF/lawSearch.do"
    params = {
        "OC": LAW_API_OC,
        "target": "licbyl",  # 별표서식 목록 API
        "type": "JSON",
        "search": 2,         # 2: 법령명으로 검색
        "query": law_name,
        "display": 100       # 최대 100개까지 결과를 받아옴
    }

    try:
        # 2. API 호출
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        attachments = data.get("licbyl", [])
        if not attachments:
            print(f"'{law_name}'에 대한 별표/서식 정보를 찾지 못했습니다.")
            return None

        # 3. 결과 필터링 및 링크 추출
        for item in attachments:
            # '별표번호'가 정확히 일치하는 항목을 찾습니다.
            if item.get("별표번호") == attachment_number:
                pdf_link = item.get("별표서식PDF파일링크")
                print(f"성공: '{item['별표명']}'의 링크를 찾았습니다.")
                return "http://www.law.go.kr" + pdf_link
        
        print(f"'{law_name}'에서 별표 '{attachment_number}'를 찾지 못했습니다.")
        return None

    except requests.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None
    except json.JSONDecodeError:
        print(f"API 응답 분석 실패: {response.text}")
        return None
    