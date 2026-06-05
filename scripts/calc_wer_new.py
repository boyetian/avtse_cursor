#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 ASR 结果目录计算 WER。

用法：
    python scripts/calc_wer_new.py
    python scripts/calc_wer_new.py --asr_dir "测试结果/测试结果新asr"
"""

import argparse
import os
import glob
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_FILE = os.path.join(BASE, "scripts", "ref.txt")
COMPUTE_WER = os.path.join(BASE, "scripts", "compute-wer.py")


def gen_hyp_index(asr_dir):
    """从 ASR 结果目录读取所有 .txt，生成 hyp 索引文件。"""
    hyp_file = os.path.join(BASE, "scripts", "hyp_新.txt")
    entries = []
    for fpath in sorted(glob.glob(os.path.join(asr_dir, "*.txt"))):
        fname = os.path.basename(fpath)
        # 文件名形如 102_out.txt → id=102
        fid = os.path.splitext(fname)[0].replace("_out", "")
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        entries.append((int(fid), fid, content))

    entries.sort(key=lambda x: x[0])

    with open(hyp_file, "w", encoding="utf-8", newline="") as out:
        for _, fid, content in entries:
            out.write(f"{fid}\t{content}\r\n")

    print(f"[1/2] 生成 hyp 索引: {hyp_file} ({len(entries)} 条)")
    return hyp_file


def run_wer(hyp_file, output_file=None):
    """运行 compute-wer.py。"""
    print(f"[2/2] 计算 WER...")
    result = subprocess.run(
        [sys.executable, COMPUTE_WER, "--char=1", "--v=1", REF_FILE, hyp_file],
        capture_output=True, text=True, cwd=os.path.dirname(COMPUTE_WER),
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # 保存到文件
    if output_file is None:
        output_file = os.path.join(BASE, "测试结果", "wer_新.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"结果已保存: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_dir", default=None, help="ASR 结果目录")
    parser.add_argument("--output", default=None, help="WER 结果输出文件 (默认: 测试结果/wer_新.txt)")
    args = parser.parse_args()

    asr_dir = args.asr_dir or os.path.join(BASE, "测试结果", "测试结果新asr")
    if not os.path.isdir(asr_dir):
        print(f"目录不存在: {asr_dir}")
        sys.exit(1)

    hyp_file = gen_hyp_index(asr_dir)
    run_wer(hyp_file, args.output)
    print("Done.")
