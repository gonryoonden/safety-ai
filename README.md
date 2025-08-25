# AI 법령정보 시스템 — 통합 개발 결과물

이 저장소에는 다음 모듈이 포함됩니다.

- utils.py: 법령 API 클라이언트, 재시도/백오프/세션 재사용, QPS 제한, get_attachment_link 구현
- vector_search_service.py: 법령 본문 수집(단일호출+JO 폴백), 파싱/정규화, 임베딩, 인덱싱(FAISS 옵션), 원자적 파일쓰기
- build_db.py: MST 리스트를 입력받아 인덱스 일괄 구축
- gemini_service.py: Gemini Function Calling 도구(get_attachment_link) 정의 및 호출 흐름
- api_server.py: /chat 엔드포인트 제공(툴 호출 결과 재주입 및 로깅)

## 환경 변수

- LAW_API_OC (필수): 법령 API OC 값
- MAX_WORKERS (기본 8): 동시 처리 워커 수
- MAX_RPS (기본 5): 전체 QPS 상한
- REQUEST_TIMEOUT (기본 15): 요청 타임아웃
- EMBEDDING_MODEL: 임베딩 모델명(현재 placeholder 해시 임베딩)
- GEMINI_API_KEY: Gemini API 키
- GEMINI_MODEL (기본 gemini-1.5-flash)

## 사용법

### 1) 인덱스 구축

```
export LAW_API_OC=your_oc
python3 build_db.py
```

또는 MST를 직접 지정:

```
MSTS=253527,266351,261457 python3 build_db.py
```

산출물: faiss_indexes/
- {MST}_answers.json
- {MST}_faiss_id_map.json
- {MST}_faiss_index.bin (FAISS 미설치 환경에서는 placeholder 내용)
- failed_{MST}.json (실패 항목이 있을 때)

### 2) 서버 실행(/chat)

```
export GEMINI_API_KEY=...
export LAW_API_OC=your_oc
python3 api_server.py
```

POST /chat
```
{
  "messages": [
    {"role": "user", "parts": [{"text": "산업안전보건기준에 관한 규칙 별표 3 링크 보여줘"}]}
  ]
}
```

응답 예시
```
{"text": "별표 3 PDF 링크: https://www.law.go.kr/LSW/flDownload.do?..."}
```

## 품질/안정성
- 세션 재사용 + 지수 백오프 재시도 + 타임아웃
- 전역 QPS 제한(RateLimiter)
- 조문 dict/list/str 수용 정규화 파서
- 파일 원자적 쓰기(.tmp 후 rename)
- MST 기반 파일명으로 Windows 호환

## 한계 및 후속 과제
- 현재 임베딩은 placeholder. 실제 임베딩 API 연동 필요
- 대용량 스트리밍 파싱은 후속 개선 지점
- 벡터 검색 API/Query는 별도 구현 필요
