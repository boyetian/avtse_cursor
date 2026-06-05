#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild wer_compare_new.csv from all wer.txt files."""

import re, csv, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(BASE, "测试结果")

def parse_wer(path):
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for m in re.finditer(r'utt: (\d+)\nWER: ([\d.]+) %', f.read()):
            d[int(m.group(1))] = float(m.group(2))
    return d

# Parse all wer files - find them by glob
import glob

wer_data = {}
for wer_path in sorted(glob.glob(os.path.join(RESULT, '*asr*', 'wer.txt'))):
    dir_name = os.path.basename(os.path.dirname(wer_path))
    # Extract short name from directory
    if 'best' in dir_name:
        name = 'best'
    elif '嘴不动' in dir_name:
        name = '嘴不动'
    elif 'new_loss' in dir_name:
        name = 'new_loss'
    elif '新asr' in dir_name:
        name = '新ASR'
    else:
        name = dir_name
    wer_data[name] = parse_wer(wer_path)
    print(f'{name}: {len(wer_data[name])} entries ({wer_path})')

# Read existing CSV for SDK/讯飞 baseline
csv_path = os.path.join(RESULT, 'wer_compare_new.csv')
old_rows = []
with open(csv_path, 'r', encoding='utf-8') as f:
    for r in csv.reader(f):
        old_rows.append(r)

# Extract baseline {id: {sdk, xf, gt_chars, gt_has}}
baseline = {}
for r in old_rows:
    if r and r[0].strip().isdigit() and len(r) >= 5:
        uid = int(r[0])
        baseline[uid] = {
            'gt_chars': r[1],
            'gt_has': r[2],
            'sdk': r[3] if r[3] else '',
            'xf': r[4] if r[4] else '',
        }

# Rebuild
all_ids = sorted(baseline.keys())
zero_ids = {57, 90, 145, 201, 202}

header = ['编号', 'GT字数', 'GT有文本', 'SDK_WER(%)', '讯飞_WER(%)',
          '新ASR_WER(%)', 'new_loss_WER(%)', '嘴不动_WER(%)', 'best_WER(%)']
rows = [header]

for uid in all_ids:
    b = baseline[uid]
    def w(name):
        if uid in zero_ids:
            return '0.00'
        v = wer_data[name].get(uid)
        return f'{v:.2f}' if v is not None else ''
    rows.append([str(uid), b['gt_chars'], b['gt_has'],
                 b['sdk'], b['xf'],
                 w('新ASR'), w('new_loss'), w('嘴不动'), w('best')])

# Stats
names = ['SDK', '讯飞', '新ASR', 'new_loss', '嘴不动', 'best']
def calc(col_idx):
    t, e = [], []
    for r in rows[1:]:
        v = r[col_idx]
        if v:
            fv = float(v)
            (t if r[2] == 'True' else e).append(fv)
    return t, e

stats = [calc(i) for i in range(3, 9)]

def avg(lst):
    return f'{sum(lst)/len(lst):.2f}' if lst else '-'

# Summary
rows.append([])
rows.append(['--- GT有文本 ---'] + [''] * 8)
for name, (t, e) in zip(names, stats):
    rows.append([name, f'{len(t)}样本', '', f'{avg(t)}'] + [''] * 5)

rows.append([])
rows.append(['--- GT为空 ---'] + [''] * 8)
for name, (t, e) in zip(names, stats):
    rows.append([name, f'{len(e)}样本', '', f'{avg(e)}'] + [''] * 5)

rows.append([])
rows.append(['--- 合计 ---'] + [''] * 8)
for name, (t, e) in zip(names, stats):
    a = t + e
    rows.append([name, f'{len(a)}样本', '', f'{avg(a)}'] + [''] * 5)

with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    csv.writer(f).writerows(rows)

# Print
tc = max(len(t) for t,_ in stats)
ec = max(len(e) for _,e in stats)
print(f'\n{"":10} {"样本":>4}  {"SDK":>8}  {"讯飞":>8}  {"新ASR":>8}  {"new_loss":>8}  {"嘴不动":>8}  {"best":>8}')
print('-' * 85)
for label, ts, es in [('GT有文本', [t for t,_ in stats], None), ('GT为空', [e for _,e in stats], None), ('合计', [t+e for t,e in stats], None)]:
    vals = '  '.join(f'{avg(v):>8}' for v in ts)
    n = len(ts[0]) if ts[0] else 0
    print(f'{label:10} {n:>4}  {vals}')
print(f'\nDone. Wrote {csv_path}')
