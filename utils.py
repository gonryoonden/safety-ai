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