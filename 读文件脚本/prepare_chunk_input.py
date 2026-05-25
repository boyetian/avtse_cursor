"""将 8 通道 PCM + MP4 视频转换为模型推理所需的格式。

8 通道 PCM 通道分配（1-indexed）：
  1, 2 — 空通道，不参与混合
  3, 4, 5, 6 — 有效通道
  7, 8 — 带噪声通道

用法：
  # 单个 chunk
  python prepare_chunk_input.py \
    --pcm chunk_001.pcm --mp4 chunk_001.mp4 \
    --mode clean --out_dir ./processed/

  # 批量处理一个目录下的 chunk 对
  python prepare_chunk_input.py \
    --pcm_dir ./pcm_chunks/ --mp4_dir ./mp4_chunks/ \
    --mode clean --out_dir ./processed/

输出文件可直接用于 main.py 的两种调用方式：
  - Case A (chunk_iter): 每个块生成独立的 npz，加载后直接喂入 process_av_stream
  - Case B (整段): 将所有块的 npz 拼接成完整 wav + frames 后再切块推理
"""

import argparse
import glob
import os

import cv2
import numpy as np
import soundfile as sf

DTYPE_MAP = {
    "int16": np.int16,
    "int32": np.int32,
    "float32": np.float32,
}


def read_8ch_pcm(pcm_path: str, sr: int, dtype: str):
    """读取 8 通道裸 PCM 文件，返回 (8, T) float32，归一化到 [-1, 1]。"""
    np_dtype = DTYPE_MAP.get(dtype)
    if np_dtype is None:
        raise ValueError(f"不支持的 dtype: {dtype}，可选: {list(DTYPE_MAP.keys())}")

    raw = np.fromfile(pcm_path, dtype=np_dtype)
    n_samples = raw.shape[0]
    if n_samples % 8 != 0:
        print(f"[warn] 总采样数 {n_samples} 不是 8 的整数倍，截断尾部")
        raw = raw[: n_samples - n_samples % 8]

    pcm = raw.reshape(-1, 8).astype(np.float32)
    if np.issubdtype(np_dtype, np.integer):
        pcm /= float(np.iinfo(np_dtype).max)

    return pcm.T, sr


def mix_channels(pcm_8ch: np.ndarray, mode: str) -> np.ndarray:
    """8 通道 PCM 混单通道。返回 (T,) float32。

    Args:
        pcm_8ch: shape (8, T)
        mode:
            clean — 平均通道 3,4,5,6（0-indexed: 2,3,4,5）
            noisy — 加权平均通道 3-8，3-6 权重 1.0，7-8 权重 0.5
    """
    if mode == "clean":
        return pcm_8ch[2:6].mean(axis=0)
    elif mode == "noisy":
        weighted = (
            pcm_8ch[2] + pcm_8ch[3] + pcm_8ch[4] + pcm_8ch[5]
            + 0.5 * pcm_8ch[6] + 0.5 * pcm_8ch[7]
        ) / 5.0
        return weighted
    else:
        raise ValueError(f"未知 mode: {mode}，可选: clean, noisy")


def read_mp4_frames(mp4_path: str):
    """读取 mp4 视频帧，返回 (frames_list, fps)。每帧为 BGR uint8 (H,W,3)。"""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {mp4_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"视频无帧: {mp4_path}")
    return frames, fps


def process_one_chunk(pcm_path: str, mp4_path: str, sr: int, dtype: str, mode: str):
    """处理单个 chunk，返回 (audio, frames, sr, fps)。

    audio: np.ndarray (1, T) float32 — 单通道，保留 C 维度供 SDK 识别
    frames: list of np.ndarray (H,W,3) uint8 BGR
    """
    pcm_8ch, sr = read_8ch_pcm(pcm_path, sr, dtype)
    wav_mono = mix_channels(pcm_8ch, mode).astype(np.float32)
    audio = wav_mono[np.newaxis, :]  # (1, T)

    frame_list, fps = read_mp4_frames(mp4_path)
    return audio, frame_list, sr, fps


def save_chunk(out_path: str, audio: np.ndarray, frames: list, sr: int, fps: float):
    """将单个 chunk 保存为 npz，加载后可直接用于 chunk_iter。"""
    frames_arr = np.stack(frames, axis=0)  # (N, H, W, 3)
    np.savez_compressed(
        out_path,
        audio=audio,
        frames=frames_arr,
        sr=np.array([sr]),
        fps=np.array([fps]),
    )


def load_chunk(npz_path: str):
    """加载 save_chunk 保存的 npz，返回 (audio, frames, sr, fps)。

    用法：
        audio, frames, sr, fps = load_chunk("chunk_001_clean.npz")
        # audio: (1, T) float32
        # frames: list of (H,W,3) uint8 BGR
        # sr: int, fps: float
    """
    data = np.load(npz_path, allow_pickle=False)
    audio = data["audio"]  # (1, T)
    frames_arr = data["frames"]  # (N, H, W, 3)
    sr = int(data["sr"][0])
    fps = float(data["fps"][0])
    frames = [frames_arr[i] for i in range(frames_arr.shape[0])]
    return audio, frames, sr, fps


def find_chunk_pairs(pcm_dir: str, mp4_dir: str, pcm_ext: str = ".pcm", mp4_ext: str = ".mp4"):
    """在两个目录中按同名匹配 PCM 和 MP4 对。返回 [(pcm_path, mp4_path), ...]。"""
    pcm_files = sorted(glob.glob(os.path.join(pcm_dir, f"*{pcm_ext}")))
    pairs = []
    for pcm_path in pcm_files:
        stem = os.path.splitext(os.path.basename(pcm_path))[0]
        mp4_path = os.path.join(mp4_dir, f"{stem}{mp4_ext}")
        if os.path.isfile(mp4_path):
            pairs.append((pcm_path, mp4_path))
        else:
            print(f"[warn] 找不到 {stem} 对应的视频，跳过")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="8 通道 PCM + MP4 → 模型输入格式")
    # 输入：单个 chunk 或批量目录
    parser.add_argument("--pcm", default=None, help="单个 8 通道 PCM 文件路径")
    parser.add_argument("--mp4", default=None, help="单个 MP4 视频路径")
    parser.add_argument("--pcm_dir", default=None, help="PCM 文件目录（批量模式）")
    parser.add_argument("--mp4_dir", default=None, help="MP4 文件目录（批量模式）")
    # 参数
    parser.add_argument("--sr", type=int, default=16000, help="采样率 (默认 16000)")
    parser.add_argument("--dtype", default="int16", choices=list(DTYPE_MAP.keys()),
                        help="PCM 数据类型 (默认 int16)")
    parser.add_argument("--mode", default="clean", choices=["clean", "noisy"],
                        help="通道混合模式: clean=只平均3-6, noisy=加权平均3-8 (默认 clean)")
    # 输出
    parser.add_argument("--out_dir", default=None, help="输出目录 (默认与 PCM 同目录)")
    args = parser.parse_args()

    # 确定处理列表
    if args.pcm and args.mp4:
        pairs = [(args.pcm, args.mp4)]
    elif args.pcm_dir and args.mp4_dir:
        pairs = find_chunk_pairs(args.pcm_dir, args.mp4_dir)
        if not pairs:
            raise RuntimeError("未找到匹配的 PCM-MP4 对")
    else:
        parser.error("请指定 --pcm + --mp4（单文件）或 --pcm_dir + --mp4_dir（批量）")

    for i, (pcm_path, mp4_path) in enumerate(pairs):
        stem = os.path.splitext(os.path.basename(pcm_path))[0]
        print(f"[{i + 1}/{len(pairs)}] 处理: {stem}")

        audio, frames, sr, fps = process_one_chunk(
            pcm_path, mp4_path, args.sr, args.dtype, args.mode,
        )
        print(f"  音频: {audio.shape}, 视频: {len(frames)} 帧, fps={fps}")

        out_dir = args.out_dir or os.path.dirname(pcm_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}_{args.mode}.npz")
        save_chunk(out_path, audio, frames, sr, fps)
        print(f"  保存: {out_path}")

    print(f"\n[完成] 共处理 {len(pairs)} 个 chunk")
    print("加载示例:")
    print("  from scripts.prepare_chunk_input import load_chunk")
    print("  audio, frames, sr, fps = load_chunk('xxx_clean.npz')")
    print("  # Case A: 直接用于 chunk_iter")
    print("  # chunk_iter.append((audio, frames))")


if __name__ == "__main__":
    main()
