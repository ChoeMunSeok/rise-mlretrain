# 11/2 기준 모델(2012 학습) 결과

역할 기준: Train 2012 / Source 2013 / Validation 2014 / Test 2015~2018. 저장소에는 `loss_monthly.csv`와 `월별_요약.csv`만 있고, 나머지 파일은 아래 재현 명령으로 만든다.

2012(Train 역할, 30,533행)로 학습한 XGBoost 한 개를 고정해 두고, 320,315행 전체를 채점해 월별·100앱 Chunk 손실표를 만들었다.

## 파일

| 파일 | 내용 | 행 수 |
|---|---|---|
| `scores.csv` | 앱별 점수·판정 (`sample_id`로 `분할표.csv`와 연결) | 320,315 |
| `loss_monthly.csv` | 월별 TP·FP·FN·TN·손실 | 73개월 × 2 view = 146 |
| `loss_chunk100.csv` | 100앱 Chunk별 TP·FP·FN·TN·손실 | 3,205 × 2 view = 6,410 |
| `chunk100_members.csv` | 어느 앱이 어느 Chunk에 들어가는지 (B가 같은 Chunk로 PAPE를 돌릴 때 사용) | 320,315 |

모델·학습 기록은 `train_xgb.py`가 `model.json`, `train_info.json`으로 저장한다.

## 열 정의

**scores.csv**: `sample_id`, `xgb_score`(악성 클래스 1의 확률, 0~1), `final_prediction`(1 = 악성 판정). 판정은 `xgb_score > 0.5` (정확히 0.5이면 정상, ADAPT 공개 코드와 같음). 점수가 정확히 0.5인 행은 0개였다. `classes_` 순서는 [0, 1].

**loss_monthly.csv**: `view`, `project_role`, `month`(2012는 `2012-XX` 한 줄), `n_apps`, `n_malicious`, `n_pred_malicious`(악성 판정 수), `TP`, `FP`, `FN`, `TN`, `loss_1to1`, `loss_2to1`, `loss_4to1`, `loss_8to1`.

**loss_chunk100.csv**: 위 열에서 `month` 대신 `chunk_id`, `ordinal`, `month_start`, `month_end`, `n_members`(Chunk에 속한 행 수), `requested_size`(100), `is_partial`, `member_hash`, `order_method`, `order_seed`가 붙는다.

- `view = all_rows`: 원행 유지 (기본). `view = excl_overlap`: 앞 역할과 겹치는 행을 뺀 결과 (보조).
- 손실 = λFN·FN + λFP·FP, λFP = 1. `loss_Nto1`은 λFN = N. 앱당 손실은 `loss / n_apps`로 계산한다. 1:1은 연구 기준선이며 실제 비용이 아니다.
- TP = 정답 악성 & 악성 판정, FP = 정답 정상 & 악성 판정, FN = 정답 악성 & 정상 판정, TN = 정답 정상 & 정상 판정.

## 100앱 Chunk 만드는 규칙 (인계서 §09)

역할(Train·Source·Validation·Test) 안에서 월 오름차순으로 놓고, 월 안에서는 정답과 무관한 순열(seed 42)로 섞은 뒤 100개씩 비중첩으로 자른다. 월을 넘어 채울 수 있다(`month_start` ≠ `month_end`). 역할 끝의 100개 미만은 `is_partial = 1`로 표시하고 다른 역할의 앱으로 채우지 않는다. Chunk 수는 Train 306 · Source 486 · Validation 585 · Test 1,828이고 각 역할의 마지막 1개가 partial이다.

`excl_overlap` view는 같은 Chunk 구성에서 겹치는 행만 뺀 것이라 `n_apps`가 100보다 작다 (3,205개 중 2,827개). Chunk끼리 비교할 때는 `loss / n_apps`를 쓴다.

## 검산 (전부 통과)

- 모든 월·Chunk에서 TP+FP+FN+TN = `n_apps`, TP+FP = 악성 판정 수, FN+TN = 정상 판정 수
- 역할별 행 수 30,533 / 48,568 / 58,485 / 182,729, 악성 3,061 / 4,854 / 5,809 / 17,080 (합 30,804 : 289,511)
- 겹침 제외 view 행 수 30,533 / 33,948 / 44,262 / 155,637 (분할표 보조 집계와 같음)
- 월별 행 수가 `월별_요약.csv` 73개월과 같음
- 월별 합 = Chunk 합 (view × 역할별 n·악성·판정·TP·FP·FN·TN)
- 모든 행이 정확히 한 Chunk에 속하고, Chunk 하나에는 한 역할만 들어 있음

## 한계

1. **Train 역할 행은 학습에 쓰인 행이다.** 그 80%는 모델을 맞추는 데, 20%는 조기종료 선택에 쓰였다. Train 행의 성능은 낙관적이므로 평가에는 Source 이후만 쓴다.
2. 앱 hash가 없어 같은 앱이 중복 포함됐는지 알 수 없다. 같은 특징 행의 반복이 많아(`dup_in_role`) 원행 유지 결과는 큰 그룹에 끌릴 수 있다. 겹침 제외 view는 앞 역할과의 중복만 뺀다.
3. 월 내부 도착 순서가 없다. 100앱 Chunk의 순서는 seed 42 순열로 만든 실험용 순서이며 초·일 단위 지연을 주장할 수 없다.
4. 2012는 월 미상(`2012-XX`)이라 월별 표에 한 줄이고, 조기종료 holdout은 시간 검증이 아니라 특징 그룹 단위 분할이다.
5. 모델은 seed 42로 1회 학습했다. 다른 seed나 하이퍼파라미터의 영향은 재지 않았다.

## 재현

```
python -I make_split_manifest.py <npz 폴더> 월별_요약.csv 분할표.csv
python -I train_xgb.py <npz 폴더> 분할표.csv models/xgb_train2012
python -I score_and_loss.py <npz 폴더> 분할표.csv 월별_요약.csv models/xgb_train2012/model.json 결과
```
