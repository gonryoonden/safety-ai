import google.generativeai as genai
import os
import logging
import json
from dotenv import load_dotenv # 👈 1. 라이브러리 임포트
from typing import List, Dict, Any

# 👈 2. FastAPI 앱을 생성하기 전에 가장 먼저 .env 파일을 로드합니다.
#    이렇게 하면 이 파일뿐만 아니라, 여기서 임포트하는 모든 다른 파일(vector_search_service 등)에서도
#    API 키를 정상적으로 사용할 수 있습니다.
load_dotenv()

# API 키 설정 (load_dotenv()를 통해 .env 파일에서 로드)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

def rerank_and_select_top_answers(question: str, results: list, top_n: int) -> list:
    """
    Gemini API를 사용하여 벡터 검색 결과를 재정렬하고 상위 N개를 선택합니다.

    Args:
        question: 사용자 질문
        results: 벡터 검색 결과 리스트 (후보군)
        top_n: 선택할 상위 답변 개수

    Returns:
        재정렬된 상위 N개의 결과 리스트
    """
    # 템플릿화된 프롬프트 구성
    # 각 문서에 번호를 매겨 Gemini가 식별하기 쉽게 함
    documents_for_prompt = ""
    for i, res in enumerate(results):
        text = res.get("data", {}).get("text", "")
        documents_for_prompt += f"--- 문서 {i+1} ---\n{text}\n\n"

    prompt = f"""
    사용자 질문: "{question}"

    아래는 위 질문과 관련하여 검색된 문서 목록입니다. 각 문서가 질문에 얼마나 정확하고 직접적인 답변을 제공하는지 평가하여, 가장 관련성이 높은 순서대로 문서 번호를 나열해주십시오. 가장 관련성 높은 문서부터 먼저, 총 {top_n}개를 선택하여 번호만 쉼표(,)로 구분하여 응답하세요.

    (예시: 3, 1, 5)

    {documents_for_prompt}
    """

    try:
        # Gemini API 호출
        response = model.generate_content(prompt)
        
        # 응답 파싱 (예: "3, 1, 5" -> [3, 1, 5])
        ranked_indices = [int(i.strip()) - 1 for i in response.text.split(',')]
        
        # 원본 result 리스트를 순위에 맞게 재정렬
        reranked_results = [results[i] for i in ranked_indices if i < len(results)]
        
        return reranked_results[:top_n]

    except Exception as e:
        # Gemini API 오류 발생 시, 그냥 원본 순서대로 반환 (Fallback)
        print(f"Gemini 재정렬 중 오류 발생: {e}")
        return results[:top_n]

def generate_answer_from_context(query: str, context: str) -> str:
    """
    주어진 컨텍스트를 기반으로 사용자 질문에 대한 답변을 생성합니다.
    """
    prompt = f"""
    당신은 산업 안전 전문가입니다. 주어진 정보를 바탕으로 사용자의 질문에 대해 명확하고 간결하게 답변하세요.

    # 정보:
    {context}

    # 질문:
    {query}

    # 답변:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API 호출 중 오류: {e}")
        return "답변을 생성하는 데 실패했습니다."

def analyze_law_text(law_text: str) -> str:
    """
    주어진 법령 텍스트를 분석하고 쉽게 해석합니다.
    """
   
def generate_structured_answer(question: str, ranked_results: list) -> dict:
    """
    재정렬된 상위 법령 정보를 바탕으로, 안전관리자에게 유용한 구조화된 답변(JSON)을 생성합니다.

    Args:
        question: 사용자 질문
        ranked_results: 재정렬된 상위 결과 리스트 (1~2개)

    Returns:
        구조화된 답변이 담긴 딕셔너리
    """
    
    # 1. 컨텍스트로 사용할 법령 원문 추출
    # 재정렬된 결과에서 source_document(또는 text) 필드 값을 정확히 가져와야 합니다.
    # 현재 `answer`와 `source_document`가 비어있는 문제를 해결하는 부분이기도 합니다.
    context = ""
    source_links = []
    source_law_names = set() # 중복 링크 방지

    for result in ranked_results:
        # 'result' 객체에 'data' 키와 그 안에 'text' 키가 있다고 가정합니다.
        # 이 구조는 vector_search_service의 반환값에 따라 달라질 수 있습니다.
        doc_text = result.get("data", {}).get("text", "")
        law_name = result.get("data", {}).get("법령명한글", "출처 미상")
        link = result.get("data", {}).get("법령상세링크", "")

        context += f"--- 법령: {law_name} ---\n{doc_text}\n\n"
        
        if law_name not in source_law_names:
            source_law_names.add(law_name)
            # utils.make_law_link와 같은 함수를 사용하여 전체 URL을 만듭니다.
            full_link = f"http://www.law.go.kr{link}" if link.startswith('/') else link
            source_links.append({"law_name": law_name, "link": full_link})

    # 2. 구조화된 JSON 생성을 위한 프롬프트 정의
    prompt = f"""
    당신은 산업안전보건법 전문가입니다. 아래의 법령 정보를 바탕으로 사용자의 질문에 대해 "안전관리자"가 현장에서 바로 사용할 수 있도록, 지정된 JSON 형식으로 답변을 생성하세요.

    # 사용자 질문:
    "{question}"

    # 참고 법령:
    {context}

    # 출력 JSON 형식 (반드시 이 구조를 준수하세요):
    {{
      "summary": "질문에 대한 핵심 내용을 2~3문장으로 요약합니다.",
      "requirements": [
        {{
          "category": "관련 규정의 대분류 (예: '구조 및 설치 기준')",
          "details": [
            {{
              "item": "준수해야 할 구체적인 항목 (예: '상부 난간대는 바닥면으로부터 90cm 이상 120cm 이하에 설치')",
              "source": "근거 법령 및 조항 (예: '산업안전보건기준에 관한 규칙 제13조제1호')"
            }}
          ]
        }}
      ],
      "source_links": []
    }}

    # 지침:
    - `summary`에는 가장 중요한 내용을 요약합니다.
    - `requirements`는 관련된 내용끼리 `category`로 묶어주세요.
    - `item`에는 안전관리자가 확인해야 할 명확한 기준(수치, 조건 등)을 제시합니다.
    - `source`에는 반드시 근거가 되는 법령명과 조항을 정확히 기재합니다.
    - `source_links`는 위 '참고 법령'에 제공된 링크 정보를 포함시키세요.
    - 만약 정보가 부족하여 답변 생성이 불가능하면, "정보 없음"이라고만 답하세요.
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        print(f"Gemini 응답: {response.text}")  # 디버깅용 출력
        
        # Gemini가 생성한 텍스트에서 JSON 부분만 추출
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        
        generated_json = json.loads(cleaned_response_text)
        
        # 생성된 JSON에 소스 링크 정보 추가
        generated_json["source_links"] = source_links
        generated_json["question"] = question
        
        return generated_json

    except Exception as e:
        print(f"구조화된 답변 생성 중 오류 발생: {e}")
        return {
            "question": question,
            "summary": "답변을 생성하는 중 오류가 발생했습니다. 원본 문서를 확인해주세요.",
            "requirements": [],
            "source_links": source_links
        }
    
def transform_query(user_query: str) -> str:
    """
    [1차 Gemini 호출]
    사용자의 일상적인 질문을 Vector DB 검색에 최적화된
    핵심 키워드나 공식적인 법률 용어로 변환합니다.

    :param user_query: 사용자의 원본 질문 (예: "타워크레인 설치할 때 지켜야 할 것들")
    :return: 변환된 검색용 질의 (예: "타워크레인 설치 조립시 안전 조치 및 작업 계획서")
    """
    
    prompt = f"""
    당신은 대한민국 산업안전법규 검색 전문가입니다.
    다음 사용자의 질문을 법률 정보 데이터베이스에서 가장 관련성 높은 문서를 찾기 위한
    검색용 키워드나 간결한 공식 질문 형태로 변환해 주십시오.
    결과는 변환된 텍스트만 간결하게 제공해 주세요.

    [사용자 질문 원본]
    "{user_query}"
    """
    
    try:
        logging.info("Gemini API 호출: 질의 변환 시작...")
        response = model.generate_content(prompt)
        transformed_query = response.text.strip()
        logging.info(f"Gemini API 호출: 질의 변환 완료. 결과: {transformed_query}")
        return transformed_query
    except Exception as e:
        logging.error(f"질의 변환 중 Gemini API 오류 발생: {e}")
        # 오류 발생 시 원본 질문을 그대로 반환하여 파이프라인이 중단되지 않도록 함
        return user_query


def generate_final_answer(original_query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    [2차 Gemini 호출]
    검색된 법령 후보군과 사용자의 원본 질문을 바탕으로,
    최종적으로 사용자에게 제공할 답변을 구조화된 JSON 형태로 생성합니다.

    :param original_query: 사용자의 원본 질문
    :param retrieved_docs: vector_search_service에서 검색된 문서 후보군 리스트
                           (각 dict는 'text', 'source_link' 등을 포함)
    :return: AI가 생성한 최종 답변 (구조화된 dict)
    """
    
    # Gemini 프롬프트에 삽입할 컨텍스트(검색된 법령 본문) 포맷팅
    context_for_generation = ""
    for i, doc in enumerate(retrieved_docs):
        context_for_generation += f"--- 문서 {i+1} (출처: {doc.get('source_link', 'N/A')}) ---\n"
        # 본문이 너무 길 경우를 대비해 일부만 사용 (예: 앞 4000자)
        context_for_generation += doc.get('text', '')[:4000]
        context_for_generation += "\n\n"

    prompt = f"""
    당신은 대한민국 산업안전 AI 어시스턴트입니다.
    주어진 [사용자 질문]에 대해, 제시된 [참고 법령 문서]들의 내용만을 근거로 하여 답변을 생성해 주십시오.
    절대로 제시된 문서에 없는 내용을 지어내서는 안 됩니다.
    답변은 반드시 아래의 JSON 형식에 맞춰 생성해야 합니다.

    [사용자 질문]
    {original_query}

    [참고 법령 문서]
    {context_for_generation}

    [출력 JSON 형식]
    {{
      "summary": "질문에 대한 핵심 내용을 1~2문장으로 요약합니다.",
      "checklist": [
        "사용자가 현장에서 구체적으로 이행해야 할 핵심 의무사항이나 절차를 목록 형태로 나열합니다.",
        "각 항목은 명확하고 간결한 문장으로 작성합니다.",
        "예: '작업 계획서를 작성하고, 그 계획에 따라 작업을 수행해야 합니다.'"
      ],
      "source_document": "답변 생성에 가장 핵심적으로 사용된 '문서 N'의 출처(source_link) 하나를 명시합니다."
    }}
    """
    
    try:
        logging.info("Gemini API 호출: 최종 답변 생성 시작...")
        response = model.generate_content(prompt)
        
        # Gemini가 반환한 텍스트에서 JSON 부분만 깔끔하게 추출
        response_text = response.text.strip()
        json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
        
        final_answer = json.loads(json_part)
        logging.info("Gemini API 호출: 최종 답변 생성 및 JSON 파싱 완료.")
        return final_answer

    except json.JSONDecodeError as e:
        logging.error(f"최종 답변 생성 후 JSON 파싱 오류: {e}\n응답 텍스트: {response_text}")
        return {"summary": "답변 생성에 실패했습니다. (JSON 형식 오류)", "checklist": [], "source_document": ""}
    except Exception as e:
        logging.error(f"최종 답변 생성 중 Gemini API 오류 발생: {e}")
        return {"summary": "답변 생성에 실패했습니다. (API 호출 오류)", "checklist": [], "source_document": ""}