#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse WER results and generate comparison CSV.

Both ASR systems are compared on the SAME set: only the IDs present in hyp_xf.txt (讯飞).
"""

import re
import os

base = r'D:\GitDocs\AV_TSE_Cursor\scripts'

def parse_wer_result(filepath):
    """Parse compute-wer.py output to get per-utt WER."""
    per_utt = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r'utt: (\d+)\nWER: ([\d.]+) % N=(\d+) C=(\d+) S=(\d+) D=(\d+) I=(\d+)'
    for m in re.finditer(pattern, content):
        uid = int(m.group(1))
        per_utt[uid] = float(m.group(2))
    return per_utt

def load_content(filepath):
    """Load {id: content} from a tab-separated index file."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            fid = int(parts[0])
            content = parts[1] if len(parts) > 1 else ''
            data[fid] = content
    return data

ref_data = load_content(os.path.join(base, 'ref.txt'))
hyp_data = load_content(os.path.join(base, 'hyp.txt'))
hyp_xf_data = load_content(os.path.join(base, 'hyp_xf.txt'))

mine_per_utt = parse_wer_result(os.path.join(base, 'wer_result_mine.txt'))
xf_per_utt = parse_wer_result(os.path.join(base, 'wer_result_xf.txt'))

# Only use IDs that exist in hyp_xf.txt (讯飞) — 113 samples
common_ids = sorted(hyp_xf_data.keys())

output_path = os.path.join(base, '..', '测试结果', 'wer_compare_new.csv')

gt_text_mine = []
gt_text_xf = []
gt_empty_mine = []
gt_empty_xf = []

with open(output_path, 'w', encoding='utf-8', newline='') as f:
    f.write('编号,GT字数,GT有文本,测试用例ASR_WER(%),讯飞ASR_WER(%)\r\n')

    for uid in common_ids:
        ref_text = ref_data.get(uid, '')
        gt_chars = len(ref_text)
        gt_has_text = gt_chars > 0

        # --- 测试用例ASR WER ---
        if uid in mine_per_utt:
            mine_wer = mine_per_utt[uid]
        elif not gt_has_text and uid in hyp_data and len(hyp_data[uid]) > 0:
            mine_wer = 100.0
        elif not gt_has_text:
            mine_wer = 0.0
        else:
            mine_wer = None

        # --- 讯飞ASR WER ---
        if uid in xf_per_utt:
            xf_wer = xf_per_utt[uid]
        elif not gt_has_text and len(hyp_xf_data[uid]) > 0:
            xf_wer = 100.0
        elif not gt_has_text:
            xf_wer = 0.0
        else:
            xf_wer = None

        mine_wer_str = f'{mine_wer:.2f}' if mine_wer is not None else ''
        xf_wer_str = f'{xf_wer:.2f}' if xf_wer is not None else ''

        if mine_wer is not None:
            if gt_has_text:
                gt_text_mine.append(mine_wer)
            else:
                gt_empty_mine.append(mine_wer)
        if xf_wer is not None:
            if gt_has_text:
                gt_text_xf.append(xf_wer)
            else:
                gt_empty_xf.append(xf_wer)

        f.write(f'{uid},{gt_chars},{gt_has_text},{mine_wer_str},{xf_wer_str}\r\n')

    # Summary
    f.write('\r\n')
    f.write('--- GT有文本 ---\r\n')
    if gt_text_mine:
        f.write(f'测试用例ASR,{len(gt_text_mine)}样本,平均WER(%),{sum(gt_text_mine)/len(gt_text_mine):.2f}\r\n')
    if gt_text_xf:
        f.write(f'讯飞ASR,{len(gt_text_xf)}样本,平均WER(%),{sum(gt_text_xf)/len(gt_text_xf):.2f}\r\n')

    f.write('\r\n')
    f.write('--- GT为空 ---\r\n')
    if gt_empty_mine:
        f.write(f'测试用例ASR,{len(gt_empty_mine)}样本,平均WER(%),{sum(gt_empty_mine)/len(gt_empty_mine):.2f}\r\n')
    if gt_empty_xf:
        f.write(f'讯飞ASR,{len(gt_empty_xf)}样本,平均WER(%),{sum(gt_empty_xf)/len(gt_empty_xf):.2f}\r\n')

    all_mine = gt_text_mine + gt_empty_mine
    all_xf = gt_text_xf + gt_empty_xf
    f.write('\r\n')
    f.write('--- 合计 ---\r\n')
    if all_mine:
        f.write(f'测试用例ASR,{len(all_mine)}样本,总平均WER(%),{sum(all_mine)/len(all_mine):.2f}\r\n')
    if all_xf:
        f.write(f'讯飞ASR,{len(all_xf)}样本,总平均WER(%),{sum(all_xf)/len(all_xf):.2f}\r\n')

print(f'Generated: {output_path}')
print(f'Total samples (common to all 3 sets): {len(common_ids)}')
print(f'GT有文本 - 测试ASR: {len(gt_text_mine)} samples, avg WER={sum(gt_text_mine)/len(gt_text_mine):.2f}%' if gt_text_mine else 'GT有文本 - 测试ASR: 0 samples')
print(f'GT有文本 - 讯飞ASR: {len(gt_text_xf)} samples, avg WER={sum(gt_text_xf)/len(gt_text_xf):.2f}%' if gt_text_xf else 'GT有文本 - 讯飞ASR: 0 samples')
print(f'GT为空 - 测试ASR: {len(gt_empty_mine)} samples, avg WER={sum(gt_empty_mine)/len(gt_empty_mine):.2f}%' if gt_empty_mine else 'GT为空 - 测试ASR: 0 samples')
print(f'GT为空 - 讯飞ASR: {len(gt_empty_xf)} samples, avg WER={sum(gt_empty_xf)/len(gt_empty_xf):.2f}%' if gt_empty_xf else 'GT为空 - 讯飞ASR: 0 samples')
print(f'合计 - 测试ASR: {len(all_mine)} samples, avg WER={sum(all_mine)/len(all_mine):.2f}%')
print(f'合计 - 讯飞ASR: {len(all_xf)} samples, avg WER={sum(all_xf)/len(all_xf):.2f}%')
