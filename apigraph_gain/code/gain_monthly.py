"""업무 1 Gain 생성: KEEP과 RETRAIN의 월별 Loss 차이로 G_real을 계산한다.

목적: VĜ(H_t) 수식을 수정할 때 reference로 쓸 실제 Gain을 만든다.

기능:
- 계산: G_real = loss_KEEP − loss_RETRAIN. 양수면 재학습이 이득이다.
- 앱당 값: 월별 앱 수가 달라 g_real_per_app도 함께 저장한다.
- 검산: 두 파일의 월, 앱 수, 악성 수(TP + FN)가 같을 때만 저장한다.

사용: python -I gain_monthly.py <keep_monthly.csv> <retrain_monthly.csv> <출력 csv>
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
KEEP, RETRAIN, OUT = sys.argv[1:4]

k = pd.read_csv(KEEP, encoding='utf-8-sig')
r = pd.read_csv(RETRAIN, encoding='utf-8-sig')
assert len(k) == 12 and k['month'].tolist() == r['month'].tolist(), '월 구성이 다름'
assert (k['n_apps'] == r['n_apps']).all(), '앱 수가 다름'
assert ((k['TP'] + k['FN']) == (r['TP'] + r['FN'])).all(), '악성 수가 다름'

gain = pd.DataFrame({'month': k['month'], 'n_apps': k['n_apps'], 'loss_keep': k['loss'], 'loss_retrain': r['loss'],
                     'g_real': k['loss'] - r['loss']})
gain['g_real_per_app'] = gain['g_real'] / gain['n_apps']
gain.to_csv(OUT, index=False, encoding='utf-8-sig')

print(gain.to_string(index=False))
print(f"평균 {gain['g_real'].mean():.1f}, 중앙값 {gain['g_real'].median():.1f}, "
      f"양수 {(gain['g_real'] > 0).sum()}/12 → {OUT}")
