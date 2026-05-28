import os
import time
from typing import Optional

import cv2
import numpy as np
import soundfile as sf

from stream_inference_SDK import StreamInferenceSDK


def _load_video_frames_bgr(mp4_path: str):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {mp4_path}")
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
        raise RuntimeError(f"video has no frames: {mp4_path}")
    return frames, fps


def run_case_a_stream_chunks(streamer: StreamInferenceSDK, chunk_iter, sr: int = 16000, fps: float = 25.0):
    """
    情况A：上游实时给对齐好的 chunk（每次一包）。
    chunk_iter 的元素为 (a_chunk, v_chunk)：
      - a_chunk: np.ndarray, shape (C,T) 或 (T,)
      - v_chunk: List[np.ndarray], 每帧 (H,W,3), BGR
    """
    results = []
    chunks = list(chunk_iter)
    total = len(chunks)
    for i, (a_chunk, v_chunk) in enumerate(chunks):
        outputs = streamer.process_av_stream(
            audio_chunk=a_chunk,
            video_chunk=v_chunk,
            is_start=(i == 0),
            is_end=(i == total - 1),
            sampling_rate=int(sr),
            fps=float(fps),
        )
        results.extend(outputs)
        print(f"[Case A] 进度: {i + 1}/{total}，本次输出 {len(outputs)} 段")
    return results


def run_case_b_full_numpy(
    streamer: StreamInferenceSDK,
    wav_np: np.ndarray,
    frames_np: list,
    chunk_ms: float,
    sr: int,
    fps: float,
    chunk_fps: Optional[float] = None,
):
    """
    情况B：上游一次性给整段 numpy（wav + frames），先按任意 chunk_ms 切块再逐包喂入。

    chunk_fps: 按音频块分配视频帧时使用的帧率；默认用模型 ref_sr，与推理对齐一致。
    fps: 仍传给 process_av_stream（overlay 等）；未指定 chunk_fps 时兼作分块帧率。
    """
    if chunk_fps is None:
        chunk_fps = float(getattr(streamer._core, "ref_sr", fps))
    chunk_fps = float(chunk_fps)
    audio_step = max(1, int(round(float(sr) * (float(chunk_ms) / 1000.0))))
    wav_arr = np.asarray(wav_np)
    if wav_arr.ndim == 1:
        wav_arr = wav_arr[np.newaxis, :]

    audio_chunk_list = [wav_arr[:, i : i + audio_step] for i in range(0, int(wav_arr.shape[1]), audio_step)]

    video_chunk_list = []
    v_pos = 0
    last_frame = frames_np[-1]
    for a_chunk in audio_chunk_list:
        dur_s = float(a_chunk.shape[1]) / float(max(1, int(sr)))
        n_frames = max(1, int(round(chunk_fps * dur_s)))
        one = []
        for _ in range(n_frames):
            if v_pos < len(frames_np):
                last_frame = frames_np[v_pos]
                one.append(last_frame)
                v_pos += 1
            else:
                one.append(last_frame)
        video_chunk_list.append(one)

    results = []
    total = min(len(audio_chunk_list), len(video_chunk_list))
    # RTF计算
    sum_process_av_stream_s = 0.0
    # RTF计算

    for i, (a_chunk, v_chunk) in enumerate(zip(audio_chunk_list[:total], video_chunk_list[:total])):
        # RTF计算
        t0 = time.perf_counter()
        # RTF计算

        outputs = streamer.process_av_stream(
            audio_chunk=a_chunk,
            video_chunk=v_chunk,
            is_start=(i == 0),
            is_end=(i == total - 1),
            sampling_rate=int(sr),
            fps=float(fps),
        )
        
        # RTF计算
        sum_process_av_stream_s += time.perf_counter() - t0
        # RTF计算

        results.extend(outputs)
        print(f"[Case B] 进度: {i + 1}/{total}，本次输出 {len(outputs)} 段")
        
    # RTF计算
    audio_dur_s = float(wav_arr.shape[1]) / float(max(1, int(sr)))
    return results, sum_process_av_stream_s, audio_dur_s
    # RTF计算

    # return results


def main():
    '''
    1) 初始化
    '''
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["torch", "onnx", "onnx_quant_dynamic", "torch_jit", "torch_jit_fp16"],
        default="torch",
        help="推理后端: torch, onnx(FP32), onnx_quant_dynamic(INT8), torch_jit, torch_jit_fp16",
    )
    parser.add_argument("--infer_chunk_ms", type=float, default=200.0, help="推理hop时长(ms)")
    parser.add_argument("--context_ms", type=float, default=100.0, help="左上下文时长(ms)")
    parser.add_argument("--max_history_ms", type=float, default=100.0, help="ring buffer最大历史(ms)")
    parser.add_argument("--use_stream_cache", type=int, default=1, help="1=full-buffer 推理, 0=逐 hop 滑窗")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="",
        help="ONNX 模型路径（type=onnx/onnx_quant_dynamic 时）；空则按 --onnx_fixed 或默认动态 onnx",
    )
    parser.add_argument(
        "--onnx_fixed",
        action="store_true",
        help="使用 checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx（定长，与默认 infer_chunk_ms 对齐）",
    )
    parser.add_argument(
        "--ref_onnx_path",
        type=str,
        default="",
        help="RKNN 拆分部署：ORT ref_encoder（灰度 4D），与 --sep_onnx_path 同时使用",
    )
    parser.add_argument(
        "--sep_onnx_path",
        type=str,
        default="",
        help="RKNN 拆分部署：ORT separator（mixture+ref_feat），与 --ref_onnx_path 同时使用",
    )
    parser.add_argument(
        "--ts_path",
        type=str,
        default="",
        help="TorchScript 路径（type=torch_jit/torch_jit_fp16）；空则按 --torch_jit_fixed 或默认 torch_jit.zip",
    )
    parser.add_argument(
        "--torch_jit_fixed",
        action="store_true",
        help="使用 checkpoints/AV_Mossformer/torch_jit_fixed.zip（定长 trace，与 av_mossformer2_fixed.onnx 同形）",
    )
    parser.add_argument(
        "--save_face_video_dir",
        type=str,
        default="",
        help="若指定目录，则保存裁出人脸 mp4（{id}_face.mp4），空=关闭",
    )
    parser.add_argument(
        "--save_face_overlay_video_dir",
        type=str,
        default="./测试结果_视频/人脸框",
        help="若指定目录，则保存原分辨率叠加框 mp4（绿=目标，黄=干扰），{id}_face_overlay.mp4；测 RTF 建议关闭",
    )
    parser.add_argument(
        "--bench_rtf",
        action="store_true",
        help="测 RTF：关闭 overlay/face 视频写出，避免编码计入 process_av_stream",
    )
    parser.add_argument(
        "--overlay_write_stride",
        type=int,
        default=15,
        help="overlay 每 N 帧写 1 帧（2=减半编码量，降低 RTF 影响）",
    )
    parser.add_argument(
        "--overlay_scale",
        type=float,
        default=0.1,
        help="overlay 输出缩放，如 0.5=半分辨率，降低编码耗时",
    )
    parser.add_argument(
        "--face_detector",
        choices=["haar", "mediapipe"],
        default="mediapipe",
        help="人脸检测后端: haar(OpenCV) 或 mediapipe",
    )
    parser.add_argument(
        "--face_detector_model",
        type=str,
        default="detector.tflite",
        help="MediaPipe 人脸模型路径（face_detector=mediapipe 时使用）",
    )
    parser.add_argument(
        "--mediapipe_lip_crop",
        default=1,
        help="face_detector=mediapipe 时以嘴部关键点为中心做参考裁剪（边长按人脸框比例缩放）",
    )
    parser.add_argument(
        "--mediapipe_lip_crop_scale",
        type=float,
        default=0.8,
        help="嘴裁正方形边长 = 该系数 × last_box 边长（图像空间），再缩放到 face_crop_size",
    )
    parser.add_argument(
        "--mediapipe_lip_crop_min_px",
        type=int,
        default=48,
        help="嘴裁在图像空间的最小边长（像素）",
    )
    parser.add_argument(
        "--mediapipe_lip_crop_max_px",
        type=int,
        default=2048,
        help="嘴裁在图像空间的最大边长（像素）",
    )
    parser.add_argument(
        "--face_target_policy",
        choices=["largest", "center", "center_largest", "center_largest_lock"],
        default="center_largest_lock",
        help="多人脸时选目标: largest/center/center_largest; center_largest_lock=居中初选后锁定",
    )
    parser.add_argument(
        "--face_target_lock",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=首帧按 policy 选人后按 IoU 锁定同一人(默认); 0=每帧重选",
    )
    parser.add_argument(
        "--face_target_lock_min_iou",
        type=float,
        default=0.15,
        help="锁定模式下与上一目标框 IoU 低于此值则不切换目标(保持上一帧)",
    )
    args = parser.parse_args()
    if args.bench_rtf:
        args.save_face_overlay_video_dir = ""
        args.save_face_video_dir = ""
        print("[bench_rtf] 已关闭 overlay/face 视频写出，RTF 仅含推理+人脸跟踪")

    sdk_kwargs = dict(
        infer_chunk_ms=args.infer_chunk_ms,
        context_ms=args.context_ms,
        max_history_ms=args.max_history_ms,
        use_stream_cache=args.use_stream_cache,
        face_detector=args.face_detector,
        face_detector_model_path=args.face_detector_model,
        mediapipe_use_lip_center_crop=1 if args.mediapipe_lip_crop else 0,
        mediapipe_lip_crop_scale=float(args.mediapipe_lip_crop_scale),
        mediapipe_lip_crop_min_px=int(args.mediapipe_lip_crop_min_px),
        mediapipe_lip_crop_max_px=int(args.mediapipe_lip_crop_max_px),
        face_target_policy=str(args.face_target_policy),
        face_target_lock=int(args.face_target_lock),
        face_target_lock_min_iou=float(args.face_target_lock_min_iou),
        overlay_write_stride=max(1, int(args.overlay_write_stride)),
        overlay_scale=float(args.overlay_scale),
    )
    if args.type in ("onnx", "onnx_quant_dynamic"):
        if args.ref_onnx_path and args.sep_onnx_path:
            streamer = StreamInferenceSDK(
                ref_onnx_path=str(args.ref_onnx_path),
                sep_onnx_path=str(args.sep_onnx_path),
                **sdk_kwargs,
            )
        else:
            if args.onnx_path:
                onnx_file = str(args.onnx_path)
            elif args.onnx_fixed:
                onnx_file = "checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx"
            elif args.type == "onnx_quant_dynamic":
                onnx_file = "checkpoints/AV_Mossformer/av_mossformer2_quant_dynamic.onnx"
            else:
                onnx_file = "checkpoints/AV_Mossformer/av_mossformer2.onnx"
            streamer = StreamInferenceSDK(onnx_path=onnx_file, **sdk_kwargs)
    elif args.type in ("torch_jit", "torch_jit_fp16"):
        if args.ts_path:
            ts_file = str(args.ts_path)
        elif args.torch_jit_fixed:
            ts_file = "checkpoints/AV_Mossformer/torch_jit_fixed.zip"
        elif args.type == "torch_jit_fp16":
            ts_file = "checkpoints/AV_Mossformer/torch_jit_FP16.zip"
        else:
            ts_file = "checkpoints/AV_Mossformer/torch_jit.zip"
        streamer = StreamInferenceSDK(ts_path=ts_file, **sdk_kwargs)
    else:
        streamer = StreamInferenceSDK(**sdk_kwargs)

    out_wav = "./测试结果"
    # out_wav = "./测试结果"
    # 2a)和2b) 二选一，推荐2a)，2b) 仅供兜底使用

    '''
    2a) 直接使用上游已准备好的整段 numpy 输入（推荐）
    - audio_np: np.ndarray, shape=(C,T) 或 (T,), dtype=float32
    - video_frames_np: list[np.ndarray], 每帧 shape=(H,W,3), dtype=uint8, BGR
    - sr: 采样率（例如 16000）
    - fps: 帧率（例如 25.0）
    '''
    # # 下面这 4 个变量请由业务侧在调用 main() 前准备好。
    # audio_np = None
    # video_frames_np = None
    # sr = 16000
    # fps = 25.0

    # if audio_np is None or video_frames_np is None:
    #     raise ValueError(
    #         "请先提供整段 numpy 输入：audio_np 与 video_frames_np。"
    #         "如果你仍想从文件读取，可参考注释中的兜底示例。"
    #     )

    # wav = np.asarray(audio_np)
    # if wav.ndim == 1:
    #     wav = wav[np.newaxis, :]  # 统一为 (C,T)
    # wav = wav.astype(np.float32, copy=False)
    # frames = list(video_frames_np)
    # sr = int(sr)
    # fps = float(fps)

    '''
    2b) 文件读取兜底示例（可按需注释）：
    '''
    audio_dir = "./测试用例/音频"
    video_dir = "./测试用例/视频"
    # audio_dir = "./测试用例/测试用例/audio_wav"
    # video_dir = "./测试用例/测试用例/video"
    # audio_dir = "./测试用例/测试用例/audio_wav/141.wav"
    # video_dir = "./测试用例/测试用例/video/141.mp4"

    out_dir = out_wav
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(audio_dir):
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    elif os.path.isfile(audio_dir):
        audio_files = [os.path.basename(audio_dir)]
        audio_dir = os.path.dirname(audio_dir) or "."
    else:
        raise FileNotFoundError(f"audio path not found: {audio_dir}")
    total_files = len(audio_files)

    for idx, audio_name in enumerate(audio_files):
        base_name = os.path.splitext(audio_name)[0]
        audio_path = os.path.join(audio_dir, audio_name)
        if os.path.isdir(video_dir):
            video_path = os.path.join(video_dir, base_name + ".mp4")
        elif os.path.isfile(video_dir):
            video_path = video_dir
        else:
            raise FileNotFoundError(f"video path not found: {video_dir}")

        if not os.path.exists(video_path):
            print(f"跳过 {audio_name}：找不到对应视频 {video_path}")
            continue

        print(f"\n===== 处理 [{idx+1}/{total_files}]: {base_name} =====")

        wav_file, sr_file = sf.read(audio_path, dtype="float32", always_2d=True)  # [T,C]
        wav = wav_file.T  # -> [C,T]
        frames, fps = _load_video_frames_bgr(video_path)
        sr = int(sr_file)

        if args.save_face_video_dir:
            face_dir = str(args.save_face_video_dir)
            os.makedirs(face_dir, exist_ok=True)
            streamer._core.save_face_video_path = os.path.join(face_dir, f"{base_name}_face.mp4")
            streamer._core._face_video_fps = float(fps)
        else:
            streamer._core.save_face_video_path = None

        if args.save_face_overlay_video_dir:
            overlay_dir = str(args.save_face_overlay_video_dir)
            os.makedirs(overlay_dir, exist_ok=True)
            streamer._core.save_face_overlay_video_path = os.path.join(
                overlay_dir, f"{base_name}_face_overlay.mp4"
            )
            streamer._core._face_video_fps = float(fps)
        else:
            streamer._core.save_face_overlay_video_path = None

        # 3b) 情况B：整段 numpy -> 任意 chunk_ms 切块 -> 逐包调用 process_av_stream
        outputs_all, sum_sdk_s, audio_dur_s = run_case_b_full_numpy(
            streamer=streamer,
            wav_np=wav,
            frames_np=frames,
            chunk_ms=100.0,
            sr=int(sr),
            fps=float(fps),
        )
        # RTF计算
        rtf = sum_sdk_s / audio_dur_s if audio_dur_s > 1e-9 else float("nan")
        print(f"RTF: {rtf:.3f}  (sum_sdk={sum_sdk_s:.3f}s, audio_dur={audio_dur_s:.3f}s)")
        # RTF计算

        if outputs_all:
            out_audio = np.concatenate(outputs_all, axis=0).astype(np.float32, copy=False)
            out_path = os.path.join(out_dir, f"{base_name}_out.wav")
            sf.write(out_path, out_audio, int(sr), subtype="PCM_16")
            print(f"已保存: {out_path}")
        else:
            print(f"{base_name}: 无输出")

    streamer.close()


if __name__ == "__main__":
    main()

