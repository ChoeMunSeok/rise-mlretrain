"""업무 1 결과 그림: KEEP·RETRAIN 성능과 G_real을 논문 형식 PNG로 저장한다.

목적: 발표 자료와 보고서에 넣을 그림을 CSV 결과에서 재현 가능하게 만든다.

기능:
- 1~4 KEEP / 5~8 RETRAIN: 연간 confusion matrix, 월별 Loss, 월별 Accuracy·F1·Recall, 월별 TP·FN·FP·TN.
- 9~10 G_real: KEEP·RETRAIN Loss 비교, 월별 G_real.
- 11~12 비교표: KEEP과 RETRAIN의 연간 지표와 월별 건수.
- 비교: KEEP과 RETRAIN의 축 범위를 같게 고정해 그림끼리 바로 비교할 수 있게 한다.
- 검산: 연간 confusion matrix 합이 57,452이고 g_real이 두 Loss의 차이일 때만 그린다.

사용: python -I plot_figures.py <2_결과 폴더> <그림 폴더>
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
RESULT, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
CORRECT, ERROR = '#dbe8f8', '#fbe1d6'
LOSS_MAX = 450  # KEEP·RETRAIN 공통 축: 두 그림을 눈으로 바로 비교하기 위함
MONTHS = list(range(1, 13))
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 9, 'axes.unicode_minus': False,
                     'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.facecolor': 'white'})


def clean(ax, grid_axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name, caption):
    fig.text(0.5, 0.005, caption, ha='center', va='bottom', fontsize=8.5, color=INK, wrap=True)
    for attempt in range(5):  # OneDrive 동기화 중에는 덮어쓰기가 잠시 OSError로 실패할 수 있다
        try:
            fig.savefig(f'{OUT}/{name}.png')
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(3)
    plt.close(fig)
    print(f'{OUT}/{name}.png')


def fig_matrix(df, tag, number, name):
    """연간 confusion matrix. 행 = 실제, 열 = 예측, [[TN, FP], [FN, TP]]. 칸에는 건수만 쓴다."""
    tp, fp, fn, tn = (int(df[c].sum()) for c in ('TP', 'FP', 'FN', 'TN'))
    assert tp + fp + fn + tn == 57452, '연간 합이 다름'
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.2)
    cells = [[('TN', tn, CORRECT), ('FP', fp, ERROR)], [('FN', fn, ERROR), ('TP', tp, CORRECT)]]
    for i, j in np.ndindex(2, 2):
        label, count, color = cells[i][j]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor='white', linewidth=2.5))
        ax.text(j, i - 0.09, label, ha='center', va='center', fontsize=9.5, color=MUTED)
        ax.text(j, i + 0.09, f'{count:,}', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect('equal')
    ax.spines[:].set_visible(False)
    ax.set_xticks([0, 1], ['정상(0)', '악성(1)'])
    ax.set_yticks([0, 1], ['정상(0)', '악성(1)'], rotation=90, va='center')
    ax.tick_params(colors=INK)
    ax.set_xlabel('예측(Predicted)')
    ax.set_ylabel('실제(True)')
    save(fig, name, f'그림 {number}. {tag}의 2015년 confusion matrix (건수).')


def fig_loss(df, tag, number, name):
    """월별 Loss = FN + FP 누적 막대. 막대 위에 월별 Loss 건수를 쓴다."""
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.21)
    ax.bar(MONTHS, df['FN'], color=BLUE, width=0.62, label=f"FN (합계 {int(df['FN'].sum()):,})", edgecolor='white', linewidth=0.8)
    ax.bar(MONTHS, df['FP'], bottom=df['FN'], color=ORANGE, width=0.62, label=f"FP (합계 {int(df['FP'].sum()):,})",
           edgecolor='white', linewidth=0.8)
    for x, value in zip(MONTHS, df['loss']):
        ax.annotate(f'{value:,}', (x, value), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8.5)
    ax.set_ylim(0, LOSS_MAX)
    ax.set_xticks(MONTHS)
    ax.set_xlabel('2015년 평가 월')
    ax.set_ylabel('Loss (건)')
    ax.legend(ncol=2, loc='upper left', fontsize=8.5, handlelength=1, columnspacing=1, borderaxespad=0.1)
    clean(ax)
    save(fig, name, f"그림 {number}. {tag}의 2015년 월별 Loss = FN + FP. 연간 합계 {int(df['loss'].sum()):,}건.")


def fig_metrics(df, tag, number, name):
    """월별 Accuracy·F1·Recall. 연간 값은 12개월 합산 confusion matrix에서 계산한다."""
    tp, fp, fn, tn = (int(df[c].sum()) for c in ('TP', 'FP', 'FN', 'TN'))
    annual = {'Accuracy': (tp + tn) / (tp + fp + fn + tn), 'F1': 2 * tp / (2 * tp + fp + fn), 'Recall': tp / (tp + fn)}
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    fig.subplots_adjust(left=0.12, right=0.83, top=0.83, bottom=0.21)
    series = [('Accuracy', 'accuracy', BLUE, 'o'), ('F1', 'f1', ORANGE, 's'), ('Recall', 'recall', AQUA, '^')]
    ends = np.array([df[c].iloc[-1] for _, c, _, _ in series], dtype=float)
    order = np.argsort(ends)
    placed = ends.copy()
    for a, b in zip(order, order[1:]):  # 끝 라벨이 겹치지 않도록 최소 간격을 둔다
        placed[b] = max(placed[b], placed[a] + 0.07)
    placed -= placed.mean() - ends.mean()  # 라벨 묶음의 중심을 선 끝에 맞춘다
    for k, (label, col, color, marker) in enumerate(series):
        ax.plot(MONTHS, df[col], color=color, marker=marker, markersize=5, linewidth=1.6,
                markeredgecolor='white', markeredgewidth=0.6, label=f'{label} (연간 {annual[label]:.3f})', clip_on=False)
        ax.text(12.25, placed[k], label, va='center', fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.7, 12.2)
    ax.set_xticks(MONTHS)
    ax.set_xlabel('2015년 평가 월')
    ax.set_ylabel('지표 값')
    clean(ax)
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8.5, columnspacing=1.4, handlelength=1.8)
    save(fig, name, f'그림 {number}. {tag}의 2015년 월별 Accuracy, F1, Recall.')


def fig_monthly_matrices(df, tag, number, name):
    """월별 TP·FN·FP·TN 꺾은선. TN과 FP의 차이가 수백 배라 로그 눈금 한 축에 함께 그린다."""
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    fig.subplots_adjust(left=0.13, right=0.92, top=0.83, bottom=0.25)
    for label, color, marker in [('TN', YELLOW, 'D'), ('TP', AQUA, '^'), ('FN', BLUE, 'o'), ('FP', ORANGE, 's')]:
        ax.plot(MONTHS, df[label], color=color, marker=marker, markersize=5, linewidth=1.6, markeredgecolor='white',
                markeredgewidth=0.6, label=f'{label} (합계 {int(df[label].sum()):,})', clip_on=False)
        ax.text(12.25, df[label].iloc[-1], label, va='center', fontsize=8.5)
    ax.set_yscale('log')
    ax.set_ylim(3, 7000)
    ax.set_yticks([10, 100, 1000], ['10', '100', '1,000'])
    ax.minorticks_off()
    ax.set_xlim(0.7, 12.2)
    ax.set_xticks(MONTHS)
    ax.set_xlabel('2015년 평가 월')
    ax.set_ylabel('건수 (로그 눈금)')
    clean(ax)
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8.5, columnspacing=1.2, handlelength=1.8)
    save(fig, name, f'그림 {number}. {tag}의 2015년 월별 confusion matrix 항목.\n세로축은 로그 눈금이고, 2015-08은 앱 수가 3,111개로 적다.')


def fig_loss_compare(g, number, name):
    """KEEP·RETRAIN 월별 Loss 비교. 두 선 사이가 G_real이다."""
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.25)
    ax.fill_between(MONTHS, g['loss_retrain'], g['loss_keep'], color=GRID, alpha=0.7, linewidth=0, label='G_real')
    ax.plot(MONTHS, g['loss_keep'], color=BLUE, marker='o', markersize=5, linewidth=1.6, markeredgecolor='white',
            markeredgewidth=0.6, label=f"KEEP (합계 {g['loss_keep'].sum():,})", clip_on=False)
    ax.plot(MONTHS, g['loss_retrain'], color=ORANGE, marker='s', markersize=5, linewidth=1.6, markeredgecolor='white',
            markeredgewidth=0.6, label=f"RETRAIN (합계 {g['loss_retrain'].sum():,})", clip_on=False)
    ax.set_ylim(0, LOSS_MAX)
    ax.set_xlim(0.7, 12.3)
    ax.set_xticks(MONTHS)
    ax.set_xlabel('2015년 평가 월')
    ax.set_ylabel('Loss (건)')
    clean(ax)
    ax.legend(ncol=3, loc='upper left', fontsize=8.5, handlelength=1.8, columnspacing=1.2, borderaxespad=0.1)
    save(fig, name, f'그림 {number}. KEEP과 RETRAIN의 2015년 월별 Loss. KEEP은 2012 학습 모델, RETRAIN은 평가 월 직전까지 학습한 모델이다.')


def fig_gain(g, number, name):
    """월별 G_real 막대. 막대마다 건수를 쓴다."""
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.25)
    ax.bar(MONTHS, g['g_real'], color=AQUA, width=0.62)
    for x, value in zip(MONTHS, g['g_real']):
        ax.annotate(f'{value:,}', (x, value), xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8.5)
    ax.axhline(g['g_real'].mean(), color=INK, linewidth=0.9, linestyle=(0, (4, 3)), label=f"평균 {g['g_real'].mean():.1f}")
    ax.set_ylim(0, LOSS_MAX)
    ax.set_xticks(MONTHS)
    ax.set_xlabel('2015년 평가 월')
    ax.set_ylabel('G_real (건)')
    clean(ax)
    ax.legend(loc='upper left', fontsize=8.5, handlelength=2.2, borderaxespad=0.1)
    save(fig, name, f"그림 {number}. 2015년 월별 G_real = loss_KEEP - loss_RETRAIN. 합계 {g['g_real'].sum():,}건.")


def fig_table(headers, rows, widths, name, caption, size, groups=(), last_is_total=False):
    """논문 형식 표: 가로줄만 쓰고 첫 열은 왼쪽, 나머지 열은 오른쪽으로 맞춘다."""
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.1)
    ax.axis('off')
    head_rows = 2 if groups else 1
    total = len(rows) + head_rows
    ax.set_xlim(0, sum(widths))
    ax.set_ylim(total, 0)
    right = np.cumsum(widths)  # 각 열의 오른쪽 끝
    left = right - np.array(widths)
    for label, first, last in groups:  # first·last 열을 묶는 상위 머리글
        ax.text((left[first] + right[last]) / 2, 0.5, label, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.plot([left[first] + 0.15, right[last] - 0.05], [1, 1], color=INK, linewidth=0.5)
    for c, h in enumerate(headers):
        x, ha = (0.1, 'left') if c == 0 else (right[c] - 0.1, 'right')
        ax.text(x, head_rows - 0.5, h, ha=ha, va='center', fontsize=9, fontweight='bold')
    for r, row in enumerate(rows):
        bold = last_is_total and r == len(rows) - 1
        for c, value in enumerate(row):
            x, ha = (0.1, 'left') if c == 0 else (right[c] - 0.1, 'right')
            ax.text(x, head_rows + r + 0.5, value, ha=ha, va='center', fontsize=9, fontweight='bold' if bold else 'normal')
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.axhline(head_rows, color=INK, linewidth=0.7)
    if last_is_total:
        ax.axhline(total - 1, color=INK, linewidth=0.7)
    ax.axhline(total, color=INK, linewidth=1.2)
    save(fig, name, caption)


def annual_table(keep, retrain):
    """연간 비교표. 차이 = RETRAIN - KEEP."""
    def annual(df):
        tp, fp, fn, tn = (int(df[c].sum()) for c in ('TP', 'FP', 'FN', 'TN'))
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Loss (FN + FP)': fn + fp, 'Accuracy': (tp + tn) / (tp + fp + fn + tn),
                'F1': 2 * tp / (2 * tp + fp + fn), 'Recall': tp / (tp + fn)}
    k, r = annual(keep), annual(retrain)
    rows = []
    for item in k:
        if isinstance(k[item], int):
            rows.append([item, f'{k[item]:,}', f'{r[item]:,}', f'{r[item] - k[item]:+,}'])
        else:
            rows.append([item, f'{k[item]:.3f}', f'{r[item]:.3f}', f'{r[item] - k[item]:+.3f}'])
    fig_table(['항목', 'KEEP', 'RETRAIN', '차이 (RETRAIN - KEEP)'], rows, [3.2, 2, 2, 3.6], '11_KEEP_RETRAIN_연간_비교표',
              f"그림 11. KEEP과 RETRAIN의 2015년 연간 비교. G_real 합계는 {k['Loss (FN + FP)'] - r['Loss (FN + FP)']:,}건이다.",
              (5.6, 3.3))


def monthly_table(keep, retrain, gain):
    """월별 비교표: 건수와 G_real."""
    columns = ['TP', 'FP', 'FN', 'TN', 'loss']
    rows = []
    for i in range(12):
        rows.append([keep['month'][i], f"{keep['n_apps'][i]:,}", *[f'{keep[c][i]:,}' for c in columns],
                     *[f'{retrain[c][i]:,}' for c in columns], f"{gain['g_real'][i]:,}"])
    rows.append(['합계', f"{keep['n_apps'].sum():,}", *[f'{keep[c].sum():,}' for c in columns],
                 *[f'{retrain[c].sum():,}' for c in columns], f"{gain['g_real'].sum():,}"])
    headers = ['평가 월', '앱 수', 'TP', 'FP', 'FN', 'TN', 'Loss', 'TP', 'FP', 'FN', 'TN', 'Loss', 'G_real']
    fig_table(headers, rows, [1.7, 1.3, 1.1, 0.9, 1.1, 1.4, 1.1, 1.1, 0.9, 1.1, 1.4, 1.1, 1.3], '12_KEEP_RETRAIN_월별_비교표',
              '그림 12. KEEP과 RETRAIN의 2015년 월별 건수 비교. G_real = KEEP Loss - RETRAIN Loss.',
              (7.4, 4.0), groups=[('KEEP', 2, 6), ('RETRAIN', 7, 11)], last_is_total=True)


keep = pd.read_csv(f'{RESULT}/keep_monthly.csv', encoding='utf-8-sig')
retrain = pd.read_csv(f'{RESULT}/retrain_monthly.csv', encoding='utf-8-sig')
gain = pd.read_csv(f'{RESULT}/gain_monthly.csv', encoding='utf-8-sig')
assert (gain['g_real'] == keep['loss'] - retrain['loss']).all(), 'g_real이 두 Loss의 차이와 다름'

for df, tag, first in [(keep, 'KEEP', 1), (retrain, 'RETRAIN', 5)]:
    fig_matrix(df, tag, first, f'{first}_{tag}_confusion_matrix')
    fig_loss(df, tag, first + 1, f'{first + 1}_{tag}_월별_Loss')
    fig_metrics(df, tag, first + 2, f'{first + 2}_{tag}_월별_지표')
    fig_monthly_matrices(df, tag, first + 3, f'{first + 3}_{tag}_월별_confusion_matrix')
fig_loss_compare(gain, 9, '9_KEEP_RETRAIN_월별_Loss')
fig_gain(gain, 10, '10_G_real')
annual_table(keep, retrain)
monthly_table(keep, retrain, gain)
