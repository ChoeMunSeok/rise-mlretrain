"""분할표(split manifest) 생성 + 검산.

사용: python -I make_split_manifest.py <npz 폴더> <월별_요약.csv> <출력 csv>
역할은 파일 이름이 아니라 observed_month(연도)로 정한다 (인계서 §03).
중복은 1,159칸이 완전히 같은 행끼리 묶어 새로 계산한다 (원본 중복 표시는 참고 열로만 둔다).
"""
import csv
import pickle
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
from numpy.lib import format as F

D, MONTHLY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ROLES = ['Train', 'Source', 'Validation', 'Test']
ROLE_OF_YEAR = {'2012': 0, '2013': 1, '2014': 2, '2015': 3, '2016': 3, '2017': 3, '2018': 3}
ALLOW = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
         ('numpy', 'ndarray'), ('numpy', 'dtype')}


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f'차단: {module}.{name}')
        return super().find_class(module, name)


def half_block(month):
    """반년 블록 이름. 2012는 월 미상이라 '2012' 한 덩어리."""
    return '2012' if month.startswith('2012') else f'{month[:4]}-H{1 if int(month[5:7]) <= 6 else 2}'


def prev_half(hb):
    y, h = int(hb[:4]), int(hb[-1])
    return f'{y}-H{h - 1}' if h == 2 else f'{y - 1}-H2'


def source_block(month, role):
    """Validation·Test 행을 Target으로 볼 때 쓰는 Source = 직전 반년 블록 (Train·Source 역할 행은 빈칸)."""
    return prev_half(half_block(month)) if role >= 2 else ''


def cross_files(name):
    """cross_split_dupes는 object 배열이라 pickle이 필요하다. numpy 복원용 클래스만 허용해서 연다."""
    with zipfile.ZipFile(f'{D}/{name}.npz') as zf, zf.open('cross_split_dupes.npy') as fh:
        v = F.read_magic(fh)
        (F.read_array_header_1_0 if v == (1, 0) else F.read_array_header_2_0)(fh)
        return NumpyOnlyUnpickler(fh).load()[:, 0]


rows, labels_of, group_id, zero_rows = [], defaultdict(set), {}, 0
for name in ['train', 'val', 'test']:
    z = np.load(f'{D}/{name}.npz', allow_pickle=False)
    packed = np.packbits(z['feature'], axis=1)
    zero_rows += int((z['feature'].sum(axis=1) == 0).sum())
    month, label, intra, cross = z['timestamp'], z['binary_label'], z['intra_split_dupes'], cross_files(name)
    for i in range(len(label)):
        g = group_id.setdefault(packed[i].tobytes(), len(group_id))
        labels_of[g].add(int(label[i]))
        rows.append((name, i, str(month[i]), ROLE_OF_YEAR[str(month[i])[:4]], g, int(label[i]),
                     int(intra[i] != -1), str(cross[i])))

first_role, in_role = {}, Counter()
for _, _, _, role, g, _, _, _ in rows:
    first_role[g] = min(first_role.get(g, 9), role)
    in_role[(role, g)] += 1

with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['sample_id', 'origin_file', 'original_row', 'observed_month', 'half_block', 'project_role',
                'source_block', 'feature_group', 'dup_in_role', 'dup_with_earlier_role', 'exclusion_reason',
                'orig_intra_dup', 'orig_cross_file'])
    for name, i, month, role, g, _, intra_dup, cross in rows:
        earlier = first_role[g] < role
        w.writerow([f'{name}.npz#{i}', f'{name}.npz', i, month, half_block(month), ROLES[role],
                    source_block(month, role), g, int(in_role[(role, g)] > 1),
                    int(earlier), f'dup_with_{ROLES[first_role[g]]}' if earlier else '', intra_dup, cross])

# ---- 검산 ----
fail = []


def chk(title, got, want):
    ok = got == want
    print(f'{"OK  " if ok else "불일치"} {title}: {got}' + ('' if ok else f'  (기대 {want})'))
    if not ok:
        fail.append(title)


n_role = Counter(r[3] for r in rows)
t_role = Counter(r[3] for r in rows if r[5] == 1)
chk('역할별 행 수 (Train/Source/Validation/Test)', [n_role[k] for k in range(4)], [30533, 48568, 58485, 182729])
chk('전체 행 수', len(rows), 320315)
chk('sample_id 유일 (한 행은 한 역할에만)', len({f'{r[0]}#{r[1]}' for r in rows}), len(rows))
chk('악성 합 / 정상 합', (sum(t_role.values()), len(rows) - sum(t_role.values())), (30804, 289511))
chk('역할별 악성 수', [t_role[k] for k in range(4)], [3061, 4854, 5809, 17080])
src = Counter(r[0] for r in rows if r[3] == 1)
chk('Source = val 전체 + test 2013-07~12', (src['val'], src['test']), (23676, 24892))
monthly = Counter(r[2] for r in rows)
with open(MONTHLY, encoding='utf-8-sig') as fh:
    csv_monthly = {r['month']: int(r['n_apps']) for r in csv.DictReader(fh)}
chk('월별_요약.csv와 월별 행 수 (73개월)', dict(monthly), csv_monthly)
HALVES = ['2012'] + [f'{y}-H{h}' for y in range(2013, 2019) for h in (1, 2)]
n_half = Counter(half_block(r[2]) for r in rows)
t_half = Counter(half_block(r[2]) for r in rows if r[5] == 1)
chk('반년 블록 행 수 (2012, 2013-H1 ~ 2018-H2)', [n_half[h] for h in HALVES],
    [30533, 23676, 24892, 29351, 29134, 29650, 27802, 29666, 26370, 15345, 11967, 18806, 23123])
chk('반년 블록 악성 수', [t_half[h] for h in HALVES], [3061, 2366, 2488, 2917, 2892, 2799, 2709, 2724, 2600, 1387, 1078, 1573, 2210])
src_ok = all((source_block(r[2], r[3]) != '') == (r[3] >= 2) and
             (r[3] < 2 or HALVES.index(source_block(r[2], r[3])) == HALVES.index(half_block(r[2])) - 1) for r in rows)
chk('source_block = 직전 반년 (Validation·Test만, Train·Source는 빈칸)', src_ok, True)
chk('첫 Validation 행(2014-H1)의 source_block', source_block('2014-01', 2), '2013-H2')
chk('고유 특징 벡터 수', len(group_id), 161153)
chk('같은 특징의 추가 행', len(rows) - len(group_id), 159162)
chk('정상·악성이 섞인 특징 그룹', sum(len(s) == 2 for s in labels_of.values()), 58)
chk('전체 0 벡터 행', zero_rows, 291)

keep = [r for r in rows if first_role[r[4]] >= r[3]]
n_keep, t_keep = Counter(r[3] for r in keep), Counter(r[3] for r in keep if r[5] == 1)
chk('보조 집계: 앞 역할과 겹치는 행 제외 후 행 수', [n_keep[k] for k in range(4)], [30533, 33948, 44262, 155637])
chk('보조 집계: 제외 후 악성 비율(%)', [round(100 * t_keep[k] / n_keep[k], 2) for k in range(4)], [10.03, 11.94, 11.27, 10.20])
print('같은 역할 안 반복 행(dup_in_role=1):', {ROLES[k]: sum(in_role[(k, r[4])] > 1 for r in rows if r[3] == k) for k in range(4)})
print('불일치 항목:', fail if fail else '없음')
sys.exit(1 if fail else 0)
