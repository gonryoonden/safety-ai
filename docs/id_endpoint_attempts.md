# ID Endpoint Attempts

일련번호(조/항/호/목/세목/별표일련번호) 포함 여부를 확인하기 위해 시도한 엔드포인트/파라미터 정리입니다.

## 시도 내역

- `lawService.do`
  - `target=law&type=JSON&MST=272927`
  - 결과: 일련번호 키 없음
- `lawService.do`
  - `target=law&type=XML&MST=272927`
  - 결과: 일련번호 키 없음

## 결론

현재 확인된 응답 형식에서는 일련번호가 제공되지 않아,
ID 집합 완전성 검증은 `id_check.status=skipped`로 유지합니다.
추후 일련번호 포함 엔드포인트가 확인되면 해당 문서를 업데이트합니다.
