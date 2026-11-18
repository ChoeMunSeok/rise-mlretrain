"""업무 1 KEEP 실험: 재학습하지 않은 모델의 2015년 월별 성능을 계산한다.

목적: RETRAIN과 비교해 G_real을 만들 때 쓰는 KEEP 쪽 값을 준비한다.

기능:
- 모델: PAPE와 같은 2012 학습 XGBoost를 불러온다. 학습은 하지 않는다.
- 판정: 2015년 앱을 채점해 점수 > 0.5이면 악성으로 판정한다.
- 계산: 월별 TP·FP·FN·TN, Loss(FN + FP), Accuracy, F1, Recall을 계산한다.
- 검산: 12개월 값이 PAPE actual_* 값과 모두 같을 때만 CSV로 저장한다.

사용: python -I keep_monthly.py <test.npz> <model.json> <monthly_comparison.csv> <출력 csv>
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
TEST_NPZ, MODEL, PAPE_CSV, OUT = sys.argv[1:5]

with np.load(TEST_NPZ, allow_pickle=False) as z:
    t = z['timestamp']
    ix = np.flatnonzero(np.char.startswith(t, '2015'))
    x, y, month = z['feature'][ix].astype(np.float32), z['binary_label'][ix].astype(int), t[ix]

booster = xgb.Booster()
booster.load_model(MODEL)
pred = (booster.predict(xgb.DMatrix(x)) > 0.5).astype(int)

rows = []
for m in sorted(set(month)):
    a, p = y[month == m], pred[month == m]
    tp, fp = int(((a == 1) & (p == 1)).sum()), int(((a == 0) & (p == 1)).sum())
    fn, tn = int(((a == 1) & (p == 0)).sum()), int(((a == 0) & (p == 0)).sum())
    rows.append({'month': str(m), 'n_apps': len(a), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                 'loss': fn + fp, 'loss_per_app': (fn + fp) / len(a), 'accuracy': (tp + tn) / len(a),
                 'f1': 2 * tp / (2 * tp + fp + fn), 'recall': tp / (tp + fn)})
keep = pd.DataFrame(rows)

# 검산: 건수는 정확히, 비율은 부동소수 오차 범위에서 일치해야 한다.
pape = pd.read_csv(PAPE_CSV, encoding='utf-8-sig')
assert len(keep) == 12 and keep['month'].tolist() == pape['observed_month'].tolist(), '월 구성이 다름'
assert keep['n_apps'].sum() == 57452, '2015 행 수가 다름'
pairs = {'TP': 'actual_tp', 'FP': 'actual_fp', 'FN': 'actual_fn', 'TN': 'actual_tn', 'loss': 'actual_loss',
         'accuracy': 'actual_accuracy', 'f1': 'actual_f1', 'recall': 'actual_recall'}
diff = np.column_stack([~np.isclose(keep[k], pape[v], rtol=1e-12, atol=1e-12) for k, v in pairs.items()])
if diff.any():
    print(keep[['month', *pairs]][diff.any(axis=1)].to_string(index=False))
    print(pape.loc[diff.any(axis=1), ['observed_month', *pairs.values()]].to_string(index=False))
    sys.exit('PAPE actual 값과 다른 달이 있어 저장하지 않았습니다.')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
keep.to_csv(OUT, index=False, encoding='utf-8-sig')
print(keep[['month', 'n_apps', 'TP', 'FP', 'FN', 'TN', 'loss', 'recall']].to_string(index=False))
print(f"합계: FN {keep['FN'].sum()}, FP {keep['FP'].sum()}, Loss {keep['loss'].sum()}")
print(f'검산 통과: PAPE actual 12개월 일치 → {OUT}')
