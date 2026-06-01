"""批量ASR处理脚本，跳过已有结果，支持并行。"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="批量ASR处理")
    parser.add_argument("--indir", default=None, help="音频目录")
    parser.add_argument("--outdir", default=None, help="输出目录")
    parser.add_argument("--host", default="192.168.88.101")
    parser.add_argument("--port", default="31366")
    parser.add_argument("--workers", type=int, default=8, help="并行数 (默认8)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有结果")
    args = parser.parse_args()

    indir = Path(args.indir or BASE_DIR.parent / "测试结果" / "测试用例结果")
    outdir = Path(args.outdir or BASE_DIR.parent / "测试结果" / "测试用例asr")
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*.wav"))
    if not files:
        print(f"没有找到 .wav 文件: {indir}")
        sys.exit(1)

    if not args.overwrite:
        files = [f for f in files if not (outdir / (f.stem + ".txt")).exists()]

    print(f"音频目录: {indir}")
    print(f"输出目录: {outdir}")
    print(f"待处理: {len(files)} 个")
    if not files:
        print("全部已处理")
        return

    client_script = BASE_DIR / "funasr_wss_client.py"
    python = sys.executable

    def run_one(f):
        subprocess.run([python, str(client_script),
                        "--host", args.host, "--port", args.port,
                        "--audio_in", str(f), "--output_dir", str(outdir)])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, files))

    print("done")


if __name__ == "__main__":
    main()
