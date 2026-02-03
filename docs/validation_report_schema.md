# Validation Report Schema

이 문서는 `validation_report.json` 및 QA 실패 리포트의 최소 스키마를 정의합니다.

## 필수 필드

- `failures` (array): 실패 항목 리스트 (각 항목은 위 필수 필드 포함)
- `failure_type` (string): 실패 유형
- `mst` (string): 법령 MST
- `crawl_ts` (string|null): 생성 시각(UTC, ISO 8601)
- `seed` (number|null): QA seed
- `source_anchor` (string|null): 유닛 고유 식별자
- `display_path_norm` (string|null): 표시 경로(정규화)
- `source_url` (string|null): 원문 URL
- `level` (string|null): 조/항/목/별표/부칙/서식 등
- `ref` (object|null): 참조 관계
  - `annex_refs` (array|null)
  - `ref_by` (array|null)
- `id_check` (object): ID 기반 검증 상태
  - `status` (string): `active` | `skipped`
  - `reason` (string|null): `skipped`인 경우 사유

## failure_type 값 목록

- `missing_required_meta`
- `header_mismatch`
- `annex_unit_missing`
- `annex_ref_by_missing`
- `annex_ref_missing`
- `exception_tag_missing`
- `search_norm_missing`
- `search_no_improvement`
- `search_target_not_found`
- `fail_fast_not_triggered`
- `law_json_missing`
- `id_missing`
- `id_extra`
- `id_duplicate`

## id_check.status=skipped 의미

온라인 상세 JSON에 일련번호(조/항/호/목/세목/별표일련번호)가 포함되지 않는 경우,
ID 집합 완전성 검증을 비활성화하고 `id_check.status=skipped`로 기록합니다.
이 경우 ID 누락/중복 검증은 수행되지 않습니다.

## 예시 (failfast_demo_272927_report.json 일부)

```json
{
  "id_check": {
    "status": "skipped",
    "reason": "source_ids_missing"
  },
  "failures": [
    {
      "failure_type": "missing_required_meta",
      "mst": "272927",
      "crawl_ts": "2026-02-02T02:07:17Z",
      "seed": 42,
      "source_anchor": "mst:272927|jo:1",
      "display_path_norm": "부칙",
      "source_url": "https://www.law.go.kr/DRF/lawService.do?target=law&MST=272927",
      "level": "부칙",
      "ref": {
        "annex_refs": null,
        "ref_by": null
      }
    }
  ]
}
```
