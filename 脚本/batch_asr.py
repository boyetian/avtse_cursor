import subprocess
import os
import sys

INPUT_DIR = "测试结果/测试用例结果"
OUTPUT_DIR = "测试结果/测试用例asr"
HOST = "192.168.88.101"
PORT = "31366"
TIMEOUT = 30  # 每个文件最多等 30 秒

input_dir = os.path.join(os.path.dirname(__file__), "..", INPUT_DIR)
output_dir = os.path.join(os.path.dirname(__file__), "..", OUTPUT_DIR)

os.makedirs(output_dir, exist_ok=True)

wav_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".wav")])

if not wav_files:
    print(f"No .wav files found in {input_dir}")
    sys.exit(1)

print(f"Found {len(wav_files)} wav files.\n")

success = 0
failed = 0

for i, wav in enumerate(wav_files, 1):
    wav_path = os.path.join(input_dir, wav)
    print(f"[{i}/{len(wav_files)}] Processing: {wav}")
    try:
        subprocess.run([
            sys.executable, "脚本/funasr_wss_client.py",
            "--host", HOST,
            "--port", str(PORT),
            "--audio_in", wav_path,
            "--output_dir", output_dir,
        ], timeout=TIMEOUT)
        success += 1
    except subprocess.TimeoutExpired:
        failed += 1
        print(f"  TIMEOUT: {wav} did not finish within {TIMEOUT}s, skipping.")
    except Exception as e:
        failed += 1
        print(f"  ERROR: {wav} - {e}")
    print()

print(f"All done. Success: {success}, Failed: {failed}")
