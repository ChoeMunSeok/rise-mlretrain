"""업무 1 RETRAIN 실험: 매월 말 재학습한 모델의 다음 달 성능을 계산한다.

목적: KEEP과 비교해 G_real을 만들 때 쓰는 RETRAIN 쪽 값을 준비한다.

기능:
- 학습 데이터: 평가 월 이전의 모든 앱을 쓴다. 예: 2015-01 평가는 2012 ~ 2014-12로 학습한다.
- 학습: 연구실 train_stage와 같은 절차로 트리 수를 정한 뒤 학습 데이터 전체로 새 모델을 만든다.
- 평가: 다음 달을 채점해 점수 > 0.5이면 악성으로 판정한다.
- 계산: 월별 TP·FP·FN·TN, Loss(FN + FP), Accuracy, F1, Recall을 계산한다.
- 범위: 평가 월은 2015-01 ~ 2015-12이며, Test(2016 이후)는 불러오지 않는다.
- 검산: 같은 절차로 2012를 학습해 트리 수 536을 재현한 뒤 시작한다.

사용: python -I retrain_monthly.py <npz 폴더> <출력 csv>
"""
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

sys.stdout.reconfigure(encoding='utf-8')
NPZ_DIR, OUT = sys.argv[1:3]
SEED = 11
PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cpu', n_estimators=1000, learning_rate=0.05,
              max_depth=5, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, scale_pos_weight=1.0,
              reg_lambda=1.0, n_jobs=2, eval_metric='logloss', early_stopping_rounds=40, random_state=SEED)


def fit_model(x, y, groups):
    """holdout·early stopping 분할로 트리 수를 정하고, 그 트리 수로 전체를 새로 학습한다."""
    dev, hold = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(x, y, groups))
    fi, vi = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1).split(x[dev], y[dev], groups[dev]))
    fit, valid = dev[fi], dev[vi]
    assert not (set(groups[fit]) & set(groups[valid]) or set(groups[dev]) & set(groups[hold])), '같은 Feature 벡터가 분할을 넘음'
    internal = xgb.XGBClassifier(**PARAMS)
    internal.fit(x[fit], y[fit], eval_set=[(x[valid], y[valid])], verbose=False)
    rounds = internal.best_iteration + 1
    final_params = {k: v for k, v in PARAMS.items() if k != 'early_stopping_rounds'}
    model = xgb.XGBClassifier(**{**final_params, 'n_estimators': rounds})
    model.fit(x, y, verbose=False)
    return model, rounds


parts = []
for name in ('train', 'val', 'test'):
    with np.load(f'{NPZ_DIR}/{name}.npz', allow_pickle=False) as z:
        t = z['timestamp']
        ix = np.flatnonzero(t <= '2015-12')
        parts.append((z['feature'][ix], z['binary_label'][ix].astype(int), t[ix]))
raw, y, month = (np.concatenate([p[i] for p in parts]) for i in range(3))
del parts
groups = np.array([hashlib.sha256(r.tobytes()).hexdigest() for r in raw])
x = raw.astype(np.float32)
del raw
assert len(x) == 195038, '2012~2015 행 수가 다름'

is_2012 = month == '2012-XX'
_, rounds_2012 = fit_model(x[is_2012], y[is_2012], groups[is_2012])
assert rounds_2012 == 536, f'2012 재현 실패: 트리 수 {rounds_2012}'
print('검산 통과: 2012 트리 수 536 재현', flush=True)

rows = []
for m in sorted(set(month[month >= '2015-01'])):
    started = time.perf_counter()
    train_mask, eval_mask = month < m, month == m
    model, rounds = fit_model(x[train_mask], y[train_mask], groups[train_mask])
    a, p = y[eval_mask], (model.predict_proba(x[eval_mask])[:, 1] > 0.5).astype(int)
    tp, fp = int(((a == 1) & (p == 1)).sum()), int(((a == 0) & (p == 1)).sum())
    fn, tn = int(((a == 1) & (p == 0)).sum()), int(((a == 0) & (p == 0)).sum())
    assert tp + fp + fn + tn == len(a), '혼동행렬 합이 앱 수와 다름'
    rows.append({'month': str(m), 'train_end': str(np.unique(month[train_mask])[-1]), 'n_train': int(train_mask.sum()),
                 'n_trees': rounds, 'n_apps': len(a), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                 'loss': fn + fp, 'loss_per_app': (fn + fp) / len(a), 'accuracy': (tp + tn) / len(a),
                 'f1': 2 * tp / (2 * tp + fp + fn), 'recall': tp / (tp + fn)})
    print(f"{m}: 학습 {rows[-1]['n_train']}행, {rounds} trees, Loss {fn + fp}, Recall {tp / (tp + fn):.3f}, "
          f'{time.perf_counter() - started:.0f}초', flush=True)

retrain = pd.DataFrame(rows)
assert len(retrain) == 12 and retrain['n_apps'].sum() == 57452, '2015 평가 구성이 다름'
assert retrain['n_train'].iloc[0] == 137586 and retrain['n_train'].iloc[-1] == 190068, '학습 행 수가 다름'
assert retrain['train_end'].iloc[0] == '2014-12' and retrain['train_end'].iloc[-1] == '2015-11', '학습 마지막 월이 다름'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
retrain.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f"합계: FN {retrain['FN'].sum()}, FP {retrain['FP'].sum()}, Loss {retrain['loss'].sum()} → {OUT}", flush=True)
