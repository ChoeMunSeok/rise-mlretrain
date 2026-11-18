# 회의 기준 PAPE 재실험 v8

- 원 APIGraph 320,315행과 이진 피처 1,159개, 원행 유지 정책을 그대로 사용한다.
- XGBoost는 2012년 30,533행으로 신규 학습한다.
- PAPE 초기 Source는 2013-2014년 107,053행, Validation은 2015년 57,452행이다.
- 달력 월 단위 12개 × seed 11/23/37 × uniform/dre = 72회다.
- 이번 실행은 2015 Validation까지만 수행한다. Test는 통합 후 담당자가 진행한다.
- 회의의 연간 Source 규칙은 2016에 2013-2015, 2017에 2013-2016, 2018에 2013-2017 누적이다.
- 2015 라벨은 Validation 선택 고정 후 공개하고, 각 이전 Test 연도는 완료 후 다음 해 1월 Source에 추가한다.
- source_schedule.py는 후속 통합용 시점 계약만 제공하며 Test 데이터 로딩·추정을 실행하지 않는다.

## 실행

```powershell
.\run_validation.ps1
```

합성 검증 → 새 모델 학습 → 초기 Source/2015 Validation export → 72회 PAPE → 파일 검증 → 인계 ZIP 순서다.
서로 독립인 월·seed·방법 추정은 worker 4개, 각 LightGBM은 CPU thread 2개로 계산한다.
저장 순서는 월·seed·방법 순서를 유지한다. 병렬 계산이 Source/Target이나 seed 사이 정보를 공유하지 않는다.
단일 worker와 병렬 worker의 수치 동등성을 합성 검증한다.

결과는 기존 13컬럼 앱 출력, 14컬럼 PAPE CSV, 전체 혼동행렬, GAIN/RCCDA 입력과 hash manifest를 포함한다.
각 Test 명령은 제공하지 않으며 실제 Test 데이터 로딩도 거부한다.
Git 추가 반영은 이번 재실험 결과 확인 이후에 진행한다. 개인 절대경로·데이터·모델·산출물은 커밋하지 않는다.
