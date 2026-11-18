"""4-2 점수·판정, 4-3 손실표(월별 + 100앱 Chunk), 4-4 검산.

사용: python -I score_and_loss.py <npz 폴더> <분할표.csv> <월별_요약.csv> <model.json> <출력 폴더>
- 점수: 320,315행 전체를 저장된 모델로 채점. 판정 = 점수 > 0.5 (정확히 0.5면 정상, ADAPT 공개 코드와 같음).
- 월별: (역할, 월)마다 TP/FP/FN/TN. 2012는 월 미상이라 '2012-XX' 한 덩어리.
- 100앱 Chunk (인계서 §09): 역할 안에서 월 오름차순 + 월 내부 label-free 순열(seed 42) 후 100개씩 비중첩,
  월을 넘어 채울 수 있고 역할 끝 잔여는 partial로 표시(다른 역할로 채우지 않음).
- view: all_rows = 원행 유지(기본), excl_overlap = 앞 역할과 겹치는 행 제외(보조, Chunk 구성은 같고 행만 뺌).
- 손실 = λFN·FN + λFP·FP, λFP=1, FN:FP = 1:1, 2:1, 4:1, 8:1.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
D, MANIFEST, MONTHLY, MODEL, OUT = sys.argv[1:6]
SEED, CHUNK = 42, 100
ROLES = ['Train', 'Source', 'Validation', 'Test']
os.makedirs(OUT, exist_ok=True)

# ---------- 입력: 분할표 순서 = train, val, test 원래 행 순서 ----------
m = pd.read_csv(MANIFEST, dtype={'sample_id': str, 'observed_month': str, 'exclusion_reason': str, 'source_block': str},
                low_memory=False)
X = np.vstack([np.load(f'{D}/{f}.npz', allow_pickle=False)['feature'] for f in ('train', 'val', 'test')]).astype(np.float32)
y = np.concatenate([np.load(f'{D}/{f}.npz', allow_pickle=False)['binary_label'] for f in ('train', 'val', 'test')]).astype(int)
assert len(m) == len(X) == len(y) == 320315
assert (m['sample_id'] == m['origin_file'] + '#' + m['original_row'].astype(str)).all()

# ---------- 4-2 점수·판정 ----------
b = xgb.Booster()
b.load_model(MODEL)
score = b.predict(xgb.DMatrix(X)).astype(np.float32)
pred = (score > 0.5).astype(int)
n_exact_half = int((score == 0.5).sum())
pd.DataFrame({'sample_id': m['sample_id'], 'xgb_score': [f'{s:.9g}' for s in score], 'final_prediction': pred}) \
    .to_csv(f'{OUT}/scores.csv', index=False)

df = pd.DataFrame({'role': pd.Categorical(m['project_role'], ROLES, ordered=True), 'month': m['observed_month'],
                   'overlap': m['dup_with_earlier_role'].astype(int), 'y': y, 'pred': pred})
df['TP'] = ((df.pred == 1) & (df.y == 1)).astype(int)
df['FP'] = ((df.pred == 1) & (df.y == 0)).astype(int)
df['FN'] = ((df.pred == 0) & (df.y == 1)).astype(int)
df['TN'] = ((df.pred == 0) & (df.y == 0)).astype(int)
df['n'] = 1
SUMS = ['n', 'y', 'pred', 'TP', 'FP', 'FN', 'TN']


def finish(t):
    t = t.rename(columns={'n': 'n_apps', 'y': 'n_malicious', 'pred': 'n_pred_malicious'})
    for r in (1, 2, 4, 8):
        t[f'loss_{r}to1'] = r * t['FN'] + t['FP']  # λFN=r, λFP=1
    return t


VIEWS = {'all_rows': df, 'excl_overlap': df[df.overlap == 0]}

# ---------- 4-3a 월별 손실표 ----------
mon = []
for v, sub in VIEWS.items():
    t = sub.groupby(['role', 'month'], observed=True)[SUMS].sum().reset_index()
    t.insert(0, 'view', v)
    mon.append(t)
mon = finish(pd.concat(mon, ignore_index=True)).rename(columns={'role': 'project_role'})
mon.to_csv(f'{OUT}/loss_monthly.csv', index=False)

# ---------- 4-3b 100앱 Chunk ----------
df['rank'] = 0
for month, idx in df.groupby('month').indices.items():
    yyyymm = 201200 if month.endswith('XX') else int(month.replace('-', ''))
    perm = np.random.default_rng([SEED, yyyymm]).permutation(len(idx))  # 정답·악성 비율과 무관한 순열
    df.loc[idx[perm], 'rank'] = np.arange(len(idx))
order = df.sort_values(['role', 'month', 'rank'], kind='stable')
order['position_in_role'] = order.groupby('role', observed=True).cumcount()
order['ordinal'] = order['position_in_role'] // CHUNK
order['position'] = order['position_in_role'] % CHUNK
order['chunk_id'] = order['role'].astype(str) + '-' + (order['ordinal'] + 1).astype(str).str.zfill(4)
df['chunk_id'] = order['chunk_id'].reindex(df.index)
df['ordinal'] = order['ordinal'].reindex(df.index)
df['position'] = order['position'].reindex(df.index)

members = pd.DataFrame({'sample_id': m['sample_id'], 'chunk_id': df['chunk_id'], 'position': df['position']})
members.sort_values(['chunk_id', 'position']).to_csv(f'{OUT}/chunk100_members.csv', index=False)

meta = order.groupby(['role', 'chunk_id'], observed=True).agg(
    ordinal=('ordinal', 'first'), month_start=('month', 'min'), month_end=('month', 'max'), n_members=('n', 'sum')).reset_index()
meta['requested_size'] = CHUNK
meta['is_partial'] = (meta['n_members'] < CHUNK).astype(int)
hashes = order.groupby('chunk_id')['position'].apply(lambda s: s.index.map(m['sample_id']).str.cat(sep='|'))
meta['member_hash'] = meta['chunk_id'].map(lambda c: hashlib.sha256(hashes[c].encode()).hexdigest()[:16])
meta['order_method'] = '월 오름차순 + 월 내부 label-free 순열'
meta['order_seed'] = SEED
ch = []
for v, sub in VIEWS.items():
    s = df.loc[sub.index].groupby('chunk_id')[SUMS].sum()
    t = meta.merge(s, left_on='chunk_id', right_index=True, how='left').fillna({k: 0 for k in SUMS})
    t[SUMS] = t[SUMS].astype(int)
    t.insert(0, 'view', v)
    ch.append(t)
ch = finish(pd.concat(ch, ignore_index=True)).rename(columns={'role': 'project_role'})
ch = ch.sort_values(['view', 'project_role', 'ordinal']).reset_index(drop=True)
ch.to_csv(f'{OUT}/loss_chunk100.csv', index=False)

# ---------- 4-4 검산 ----------
fail = []


def chk(title, got, want):
    ok = got == want
    print(f'{"OK  " if ok else "불일치"} {title}: {got}' + ('' if ok else f'  (기대 {want})'))
    if not ok:
        fail.append(title)


def role_sum(t, view, col):
    s = t[t['view'] == view].groupby('project_role', observed=True)[col].sum()
    return [int(s.get(r, 0)) for r in ROLES]


chk('점수 행 수 / 점수가 0~1 안', (len(score), bool(((score >= 0) & (score <= 1)).all())), (320315, True))
print(f'정보: 점수가 정확히 0.5인 행 {n_exact_half}개 (판정은 > 0.5 규칙)')
for name, t in (('월별', mon), ('Chunk', ch)):
    chk(f'{name}: TP+FP+FN+TN = n_apps (모든 행)', bool((t.TP + t.FP + t.FN + t.TN == t.n_apps).all()), True)
    chk(f'{name}: TP+FP = 악성 판정 수, FN+TN = 정상 판정 수', bool(((t.TP + t.FP == t.n_pred_malicious) &
                                                                    (t.FN + t.TN == t.n_apps - t.n_pred_malicious)).all()), True)
    chk(f'{name}: 역할별 행 수 (원행 유지)', role_sum(t, 'all_rows', 'n_apps'), [30533, 48568, 58485, 182729])
    chk(f'{name}: 역할별 악성 수 (원행 유지)', role_sum(t, 'all_rows', 'n_malicious'), [3061, 4854, 5809, 17080])
    chk(f'{name}: 역할별 행 수 (겹침 제외)', role_sum(t, 'excl_overlap', 'n_apps'), [30533, 33948, 44262, 155637])
chk('전체 악성 / 정상', (int(y.sum()), int((1 - y).sum())), (30804, 289511))
with open(MONTHLY, encoding='utf-8-sig') as fh:
    csv_monthly = {r['month']: int(r['n_apps']) for r in csv.DictReader(fh)}
mm = mon[mon['view'] == 'all_rows']
chk('월별_요약.csv와 월별 행 수 (73개월)', dict(zip(mm.month, mm.n_apps)), csv_monthly)
REN = {'n_apps': 'n', 'n_malicious': 'y', 'n_pred_malicious': 'pred'}
pooled_m = mon.rename(columns=REN).groupby(['view', 'project_role'], observed=True)[SUMS].sum()
pooled_c = ch.rename(columns=REN).groupby(['view', 'project_role'], observed=True)[SUMS].sum()
chk('월별 / Chunk pooled count 일치 (view x 역할 x n·악성·판정·TP·FP·FN·TN)', bool(pooled_m.equals(pooled_c)), True)
a = ch[ch['view'] == 'all_rows']
chk('Chunk 수 / partial 수 (Train·Source·Validation·Test)',
    ([int((a.project_role == r).sum()) for r in ROLES], [int(a[a.project_role == r].is_partial.sum()) for r in ROLES]),
    ([306, 486, 585, 1828], [1, 1, 1, 1]))
chk('partial 아닌 Chunk는 모두 정확히 100앱', bool((a[a.is_partial == 0].n_apps == 100).all()), True)
chk('Chunk 멤버십: 모든 행이 정확히 한 Chunk에 속함', (int(members.chunk_id.notna().all()), int(members.sample_id.nunique())), (1, 320315))
chk('Chunk 안에서 한 역할만 섞임', bool((order.groupby('chunk_id', observed=True)['role'].nunique() == 1).all()), True)
print('불일치 항목:', fail if fail else '없음')

# ---------- 참고: 연도별 성능 (원행 유지) ----------
df['year'] = df['month'].str[:4]
yr = df.groupby('year')[SUMS].sum()
yr['recall'] = yr.TP / (yr.TP + yr.FN)
yr['precision'] = yr.TP / (yr.TP + yr.FP)
yr['loss_1to1_per_app'] = (yr.FN + yr.FP) / yr.n
yr['loss_8to1_per_app'] = (8 * yr.FN + yr.FP) / yr.n
print(yr[['n', 'y', 'TP', 'FP', 'FN', 'TN', 'recall', 'precision', 'loss_1to1_per_app', 'loss_8to1_per_app']].round(4).to_string())
rl = df.groupby('role', observed=True)[SUMS].sum()
rl['recall'] = rl.TP / (rl.TP + rl.FN)
rl['precision'] = rl.TP / (rl.TP + rl.FP)
print(rl[['n', 'y', 'TP', 'FP', 'FN', 'TN', 'recall', 'precision']].round(4).to_string())
sys.exit(1 if fail else 0)
