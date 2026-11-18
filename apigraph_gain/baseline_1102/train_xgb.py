"""4-1 기준 모델 학습: Train 역할(2012, train.npz 30,533행)로 XGBoost를 학습한다.

사용: python -I train_xgb.py <npz 폴더> <분할표.csv> <출력 폴더>
인계서 §07: XGBClassifier, objective=binary:logistic, CPU hist, 입력 float32 0/1,
            2012 내부 그룹 holdout으로 조기종료, 트리 수·설정·seed 기록.
인계서에 값이 없는 것(learning_rate, max_depth, 트리 상한, patience, holdout 비율)은 아래에서 직접 정했다.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import sklearn
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
D, MANIFEST, OUT = sys.argv[1:4]
SEED = 42
PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cpu', n_estimators=1000, learning_rate=0.1,
              max_depth=6, eval_metric='logloss', early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
os.makedirs(OUT, exist_ok=True)

z = np.load(f'{D}/train.npz', allow_pickle=False)
X_u8, y = z['feature'], z['binary_label'].astype(int)
X = X_u8.astype(np.float32)

# 같은 특징(1,159칸 동일) 행은 같은 그룹 번호 -> 그룹 단위로 holdout을 뗀다 (분할표의 feature_group).
grp = np.full(len(y), -1)
with open(MANIFEST, encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        if r['origin_file'] == 'train.npz':
            grp[int(r['original_row'])] = int(r['feature_group'])
assert (grp >= 0).all() and len(y) == 30533

# 5-fold 중 첫 fold = 약 20%. 악성 비율을 맞추고, 같은 그룹은 한쪽에만 둔다.
tr, ho = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, grp))
assert not set(grp[tr]) & set(grp[ho]), '같은 특징 그룹이 학습·holdout 양쪽에 있음'

t0 = time.time()
clf = xgb.XGBClassifier(**PARAMS)
clf.fit(X[tr], y[tr], eval_set=[(X[ho], y[ho])], verbose=False)
secs = time.time() - t0

best = int(clf.best_iteration)
booster = clf.get_booster()[:best + 1]  # best_iteration까지의 트리만 저장 (뒤 patience 50개는 버림)
n_trees = len(booster.get_dump())
model_path = f'{OUT}/model.json'
booster.save_model(model_path)

# 저장한 모델을 다시 읽어도 같은 점수가 나오는지 확인
b2 = xgb.Booster()
b2.load_model(model_path)
p_ho = b2.predict(xgb.DMatrix(X[ho]))
assert np.allclose(p_ho, clf.predict_proba(X[ho], iteration_range=(0, best + 1))[:, 1], atol=1e-6)
assert list(clf.classes_) == [0, 1]

pred = p_ho > 0.5
yh = y[ho]
info = {
    'seed': SEED, 'params': {k: v for k, v in PARAMS.items()},
    'classes_': [int(c) for c in clf.classes_], 'threshold': 0.5, 'comparator': '>',
    'train_rows_total': int(len(y)), 'fit_rows': int(len(tr)), 'holdout_rows': int(len(ho)),
    'fit_malicious': int(y[tr].sum()), 'holdout_malicious': int(yh.sum()),
    'fit_groups': int(len(set(grp[tr]))), 'holdout_groups': int(len(set(grp[ho]))),
    'best_iteration': best, 'n_trees': n_trees, 'best_holdout_logloss': float(clf.best_score),
    'holdout_auc': float(roc_auc_score(yh, p_ho)), 'holdout_logloss_check': float(log_loss(yh, p_ho)),
    'holdout_at_0.5': {'TP': int((pred & (yh == 1)).sum()), 'FP': int((pred & (yh == 0)).sum()),
                       'FN': int((~pred & (yh == 1)).sum()), 'TN': int((~pred & (yh == 0)).sum())},
    'feature_count': int(X.shape[1]), 'feature_order': 'NPZ 열 0..1158 그대로 (특징 이름 없음)',
    'train_matrix_sha256': hashlib.sha256(X_u8.tobytes()).hexdigest(),
    'model_sha256': hashlib.sha256(open(model_path, 'rb').read()).hexdigest(),
    'train_seconds': round(secs, 1), 'xgboost': xgb.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__,
}
json.dump(info, open(f'{OUT}/train_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(info, ensure_ascii=False, indent=2))
