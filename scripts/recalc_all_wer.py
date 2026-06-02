#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalculate all 5 WER CSV files using the updated compute-wer.py.

Runs compute-wer.py --char=1 for both ASR systems, parses per-utterance WER,
then generates all 5 CSV comparison files with exact format fidelity.
"""

import os
import re
import subprocess
import sys
import csv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(BASE_DIR, '测试结果')

REF_FILE = os.path.join(SCRIPT_DIR, 'ref.txt')
HYP_MINE_FILE = os.path.join(SCRIPT_DIR, 'hyp.txt')
HYP_XF_FILE = os.path.join(SCRIPT_DIR, 'hyp_xf.txt')

OUTPUTS = {
    'wer_result':        os.path.join(RESULT_DIR, 'wer_result.csv'),
    'wer_compare_full':  os.path.join(RESULT_DIR, 'wer_compare_full.csv'),
    'wer_mine_full':     os.path.join(RESULT_DIR, 'wer_mine_full.csv'),
    'wer_compare_new':   os.path.join(RESULT_DIR, 'wer_compare_new.csv'),
    'wer_compare':       os.path.join(RESULT_DIR, 'wer_compare.csv'),
}


def load_data(filepath):
    """Load tab-separated file {id: transcript}."""
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


def run_compute_wer(hyp_file):
    """Run compute-wer.py and parse per-utt WER from stdout. Returns {id: float}."""
    result = subprocess.run(
        [sys.executable, 'compute-wer.py', '--char=1', '--v=1', 'ref.txt', os.path.basename(hyp_file)],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
        encoding='utf-8', errors='replace'
    )
    if result.returncode != 0:
        print(f"Error running compute-wer.py on {hyp_file}:")
        print(result.stderr)
        sys.exit(1)

    per_utt = {}
    pattern = r'utt: (\d+)\nWER: ([\d.]+) % N=(\d+) C=(\d+) S=(\d+) D=(\d+) I=(\d+)'
    for m in re.finditer(pattern, result.stdout):
        uid = int(m.group(1))
        per_utt[uid] = float(m.group(2))
    return per_utt


def avg_or_empty(values):
    """Average of list rounded to 2 decimal places, or empty string if list empty."""
    if not values:
        return ''
    return round(sum(values) / len(values), 2)


def format_wer(val, style='float'):
    """Format WER value. 'float' = Python repr, 'fixed2' = 2 decimal places."""
    if val is None:
        return ''
    v = round(val, 2)
    if style == 'fixed2':
        return f'{v:.2f}'
    return str(v)


def build_records(ref_data, hyp_mine_data, hyp_xf_data, mine_wer, xf_wer):
    """Build unified records for all common IDs."""
    common_ids = sorted(hyp_xf_data.keys())
    records = []
    gt_text_mine, gt_text_xf = [], []
    gt_empty_mine, gt_empty_xf = [], []

    for uid in common_ids:
        ref_text = ref_data.get(uid, '')
        gt_chars = len(ref_text)
        gt_has_text = gt_chars > 0
        mine_hyp = hyp_mine_data.get(uid, '')
        xf_hyp = hyp_xf_data.get(uid, '')

        # Resolve mine WER — use raw compute-wer.py output for all cases
        if uid in mine_wer:
            m_wer = mine_wer[uid]
        elif not gt_has_text and len(mine_hyp) > 0:
            m_wer = 100.0
        elif not gt_has_text:
            m_wer = 0.0
        else:
            m_wer = None

        # Resolve xf WER
        if uid in xf_wer:
            x_wer = xf_wer[uid]
        elif not gt_has_text and len(xf_hyp) > 0:
            x_wer = 100.0
        elif not gt_has_text:
            x_wer = 0.0
        else:
            x_wer = None

        rec = {
            'id': uid,
            'gt_chars': gt_chars,
            'gt_has_text': gt_has_text,
            'mine_wer': m_wer,
            'xf_wer': x_wer,
            'mine_hyp_chars': len(mine_hyp),
            'xf_hyp_chars': len(xf_hyp),
        }
        records.append(rec)

        if m_wer is not None:
            if gt_has_text:
                gt_text_mine.append(m_wer)
            else:
                gt_empty_mine.append(m_wer)
        if x_wer is not None:
            if gt_has_text:
                gt_text_xf.append(x_wer)
            else:
                gt_empty_xf.append(x_wer)

    return records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf


# ── CSV Generators ──────────────────────────────────────────────────────

def gen_wer_result(records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf):
    """wer_result.csv: 编号, GT字数, 测试用例ASR_WER(%), 讯飞ASR_WER(%)"""
    path = OUTPUTS['wer_result']
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['编号', 'GT字数', '测试用例ASR_WER(%)', '讯飞ASR_WER(%)'])
        for r in records:
            w.writerow([r['id'], r['gt_chars'], format_wer(r['mine_wer']), format_wer(r['xf_wer'])])

        w.writerow([])
        w.writerow(['统计', '', '', ''])
        w.writerow(['-- GT != 0 (有参考文本) --', '', '', ''])
        w.writerow(['测试用例ASR 有效样本', len(gt_text_mine), '平均WER(%)', avg_or_empty(gt_text_mine)])
        w.writerow(['测试用例ASR 无样本', 0, '', ''])
        w.writerow(['讯飞ASR 有效样本', len(gt_text_xf), '平均WER(%)', avg_or_empty(gt_text_xf)])
        w.writerow(['讯飞ASR 无样本', 0, '', ''])
        w.writerow([])
        w.writerow(['-- GT == 0 (空参考) --', '', '', ''])
        w.writerow(['测试用例ASR 有效样本', len(gt_empty_mine), '平均WER(%)', avg_or_empty(gt_empty_mine)])
        w.writerow(['测试用例ASR 无样本', 0, '', ''])
        w.writerow(['讯飞ASR 有效样本', len(gt_empty_xf), '平均WER(%)', avg_or_empty(gt_empty_xf)])
        w.writerow(['讯飞ASR 无样本', 0, '', ''])
        w.writerow([])
        total1 = len(gt_text_mine) + len(gt_empty_mine)
        total2 = len(gt_text_xf) + len(gt_empty_xf)
        w.writerow(['-- 合计 --', '', '', ''])
        w.writerow(['测试用例ASR 总有效样本', total1, '总平均WER(%)', avg_or_empty(gt_text_mine + gt_empty_mine)])
        w.writerow(['讯飞ASR 总有效样本', total2, '总平均WER(%)', avg_or_empty(gt_text_xf + gt_empty_xf)])
        w.writerow(['总编号数', len(records), '', ''])
    print(f'  [OK] {os.path.basename(path)} ({len(records)} rows)')


def gen_wer_compare_full(records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf):
    """wer_compare_full.csv: 编号, GT字数, GT有文本, 测试用例ASR_WER(%), 讯飞ASR_WER(%)"""
    path = OUTPUTS['wer_compare_full']
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['编号', 'GT字数', 'GT有文本', '测试用例ASR_WER(%)', '讯飞ASR_WER(%)'])
        for r in records:
            w.writerow([r['id'], r['gt_chars'], r['gt_has_text'], format_wer(r['mine_wer']), format_wer(r['xf_wer'])])

        w.writerow([])
        w.writerow(['--- GT有文本 ---', '', '', '', ''])
        w.writerow(['测试用例ASR', f'{len(gt_text_mine)}样本', '平均WER(%)', f'{avg_or_empty(gt_text_mine):.2f}' if gt_text_mine else '', ''])
        w.writerow(['讯飞ASR', f'{len(gt_text_xf)}样本', '平均WER(%)', f'{avg_or_empty(gt_text_xf):.2f}' if gt_text_xf else '', ''])
        w.writerow([])
        w.writerow(['--- GT为空 ---', '', '', '', ''])
        w.writerow(['测试用例ASR', f'{len(gt_empty_mine)}样本', '平均WER(%)', f'{avg_or_empty(gt_empty_mine):.2f}' if gt_empty_mine else '', ''])
        w.writerow(['讯飞ASR', f'{len(gt_empty_xf)}样本', '平均WER(%)', f'{avg_or_empty(gt_empty_xf):.2f}' if gt_empty_xf else '', ''])
        all_mine = gt_text_mine + gt_empty_mine
        all_xf = gt_text_xf + gt_empty_xf
        w.writerow([])
        w.writerow(['--- 合计 ---', '', '', '', ''])
        w.writerow(['测试用例ASR', f'{len(all_mine)}样本', '总平均WER(%)', f'{avg_or_empty(all_mine):.2f}' if all_mine else '', ''])
        w.writerow(['讯飞ASR', f'{len(all_xf)}样本', '总平均WER(%)', f'{avg_or_empty(all_xf):.2f}' if all_xf else '', ''])
    print(f'  [OK] {os.path.basename(path)} ({len(records)} rows)')


def gen_wer_mine_full(records, ref_data, hyp_mine_data, gt_text_mine, gt_empty_mine):
    """wer_mine_full.csv: 编号, GT字数, 测试用例ASR_WER(%), 备注"""
    path = OUTPUTS['wer_mine_full']
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['编号', 'GT字数', '测试用例ASR_WER(%)', '备注'])
        for r in records:
            if not r['gt_has_text'] and r['mine_hyp_chars'] > 0:
                note = f'GT空, ASR={r["mine_hyp_chars"]}字'
            elif not r['gt_has_text']:
                note = 'GT空, ASR=0字'
            else:
                note = ''
            w.writerow([r['id'], r['gt_chars'], format_wer(r['mine_wer']), note])

        w.writerow([])
        w.writerow(['--- GT有文本 ---', '', '', ''])
        w.writerow(['有效样本', len(gt_text_mine), '平均WER(%)', f'{avg_or_empty(gt_text_mine):.2f}' if gt_text_mine else ''])
        w.writerow(['无样本', 0, '', ''])
        w.writerow([])
        w.writerow(['--- GT为空 ---', '', '', ''])
        w.writerow(['有效样本', len(gt_empty_mine), '平均WER(%)', f'{avg_or_empty(gt_empty_mine):.2f}' if gt_empty_mine else ''])
        w.writerow(['无样本', 0, '', ''])
        all_mine = gt_text_mine + gt_empty_mine
        w.writerow([])
        w.writerow(['--- 合计 ---', '', '', ''])
        w.writerow(['总有效样本', len(all_mine), '总平均WER(%)', f'{avg_or_empty(all_mine):.2f}' if all_mine else ''])
    print(f'  [OK] {os.path.basename(path)} ({len(records)} rows)')


def gen_wer_compare_new(records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf):
    """wer_compare_new.csv: matches gen_wer_compare.py output format exactly."""
    path = OUTPUTS['wer_compare_new']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('编号,GT字数,GT有文本,测试用例ASR_WER(%),讯飞ASR_WER(%)\r\n')
        for r in records:
            m = format_wer(r['mine_wer'], 'fixed2')
            x = format_wer(r['xf_wer'], 'fixed2')
            f.write(f'{r["id"]},{r["gt_chars"]},{r["gt_has_text"]},{m},{x}\r\n')

        f.write('\r\n')
        f.write('--- GT有文本 ---\r\n')
        if gt_text_mine:
            f.write(f'测试用例ASR,{len(gt_text_mine)}样本,平均WER(%),{avg_or_empty(gt_text_mine):.2f}\r\n')
        if gt_text_xf:
            f.write(f'讯飞ASR,{len(gt_text_xf)}样本,平均WER(%),{avg_or_empty(gt_text_xf):.2f}\r\n')

        f.write('\r\n')
        f.write('--- GT为空 ---\r\n')
        if gt_empty_mine:
            f.write(f'测试用例ASR,{len(gt_empty_mine)}样本,平均WER(%),{avg_or_empty(gt_empty_mine):.2f}\r\n')
        if gt_empty_xf:
            f.write(f'讯飞ASR,{len(gt_empty_xf)}样本,平均WER(%),{avg_or_empty(gt_empty_xf):.2f}\r\n')

        all_mine = gt_text_mine + gt_empty_mine
        all_xf = gt_text_xf + gt_empty_xf
        f.write('\r\n')
        f.write('--- 合计 ---\r\n')
        if all_mine:
            f.write(f'测试用例ASR,{len(all_mine)}样本,总平均WER(%),{avg_or_empty(all_mine):.2f}\r\n')
        if all_xf:
            f.write(f'讯飞ASR,{len(all_xf)}样本,总平均WER(%),{avg_or_empty(all_xf):.2f}\r\n')
    print(f'  [OK] {os.path.basename(path)} ({len(records)} rows)')


def gen_wer_compare(records, gt_text_mine, gt_text_xf):
    """wer_compare.csv: only GT-has-text rows. 编号, 测试用例ASR_WER(%), 讯飞ASR_WER(%), WER差值"""
    path = OUTPUTS['wer_compare']
    gt_records = [r for r in records if r['gt_has_text']]
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['编号', '测试用例ASR_WER(%)', '讯飞ASR_WER(%)', 'WER差值'])
        for r in gt_records:
            m = format_wer(r['mine_wer'])
            x = format_wer(r['xf_wer'])
            diff = round(r['mine_wer'] - r['xf_wer'], 2) if r['mine_wer'] is not None and r['xf_wer'] is not None else ''
            w.writerow([r['id'], m, x, str(diff)])

        w.writerow([])
        w.writerow(['统计', '', '', ''])
        w.writerow(['测试用例ASR 有效样本', len(gt_text_mine), '平均WER(%)', avg_or_empty(gt_text_mine)])
        w.writerow(['讯飞ASR 有效样本', len(gt_text_xf), '平均WER(%)', avg_or_empty(gt_text_xf)])
        w.writerow(['测试用例ASR 无样本', 0, '', ''])
    print(f'  [OK] {os.path.basename(path)} ({len(gt_records)} rows)')


def main():
    print("=" * 60)
    print("Recalculating all WER CSV files")
    print("=" * 60)

    # Phase 1: Load source data
    print("\n[Phase 1] Loading source data...")
    ref_data = load_data(REF_FILE)
    hyp_mine_data = load_data(HYP_MINE_FILE)
    hyp_xf_data = load_data(HYP_XF_FILE)
    print(f"  ref.txt: {len(ref_data)} entries")
    print(f"  hyp.txt: {len(hyp_mine_data)} entries")
    print(f"  hyp_xf.txt: {len(hyp_xf_data)} entries")

    # Phase 2: Run compute-wer.py
    print("\n[Phase 2] Running compute-wer.py --char=1...")
    print("  → ref.txt vs hyp.txt (mine)...")
    mine_wer = run_compute_wer(HYP_MINE_FILE)
    print(f"    {len(mine_wer)} utterances with WER results")

    print("  → ref.txt vs hyp_xf.txt (xf)...")
    xf_wer = run_compute_wer(HYP_XF_FILE)
    print(f"    {len(xf_wer)} utterances with WER results")

    # Phase 3: Build unified records
    print("\n[Phase 3] Building unified data model...")
    records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf = \
        build_records(ref_data, hyp_mine_data, hyp_xf_data, mine_wer, xf_wer)
    print(f"  Total common IDs: {len(records)}")
    print(f"  GT有文本: mine={len(gt_text_mine)}, xf={len(gt_text_xf)}")
    print(f"  GT为空:   mine={len(gt_empty_mine)}, xf={len(gt_empty_xf)}")

    # Phase 4: Generate all 5 CSV files
    print("\n[Phase 4] Generating CSV files...")
    gen_wer_result(records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf)
    gen_wer_compare_full(records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf)
    gen_wer_mine_full(records, ref_data, hyp_mine_data, gt_text_mine, gt_empty_mine)
    gen_wer_compare_new(records, gt_text_mine, gt_text_xf, gt_empty_mine, gt_empty_xf)
    gen_wer_compare(records, gt_text_mine, gt_text_xf)

    # Phase 5: Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nTotal IDs processed: {len(records)}")
    print(f"\n--- GT有文本 ---")
    if gt_text_mine:
        print(f"  测试用例ASR: {len(gt_text_mine)} samples, avg WER = {avg_or_empty(gt_text_mine):.2f}%")
    if gt_text_xf:
        print(f"  讯飞ASR:     {len(gt_text_xf)} samples, avg WER = {avg_or_empty(gt_text_xf):.2f}%")
    print(f"\n--- GT为空 ---")
    if gt_empty_mine:
        print(f"  测试用例ASR: {len(gt_empty_mine)} samples, avg WER = {avg_or_empty(gt_empty_mine):.2f}%")
    if gt_empty_xf:
        print(f"  讯飞ASR:     {len(gt_empty_xf)} samples, avg WER = {avg_or_empty(gt_empty_xf):.2f}%")
    all_mine = gt_text_mine + gt_empty_mine
    all_xf = gt_text_xf + gt_empty_xf
    print(f"\n--- 合计 ---")
    print(f"  测试用例ASR: {len(all_mine)} samples, avg WER = {avg_or_empty(all_mine):.2f}%")
    print(f"  讯飞ASR:     {len(all_xf)} samples, avg WER = {avg_or_empty(all_xf):.2f}%")
    print(f"\nAll files written to: {RESULT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
