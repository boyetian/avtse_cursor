import os
import time

import cv2
import numpy as np
import soundfile as sf

from stream_inference_SDK import StreamInferenceSDK, StreamProcessor
from visual_preprocessor import VisualPreprocessor


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
        "--chunk_ms",
        type=float,
        default=100.0,
        help="输入切片粒度(ms)，测试模式下按此粒度切分整段文件模拟流式",
    )
    parser.add_argument(
        "--write_lip",
        action="store_true",
        help="输出裁剪后的嘴唇/人脸视频",
    )
    parser.add_argument(
        "--write_lip_dir",
        type=str,
        default="./测试结果_嘴唇视频",
        help="--write_lip 输出目录",
    )
    parser.add_argument(
        "--face_scale",
        type=float,
        default=1.0,
        help="人脸框缩放系数 (默认 1.0)",
    )
    parser.add_argument(
        "--detect_every_n",
        type=int,
        default=5,
        help="人脸检测间隔帧数，越小跟踪越紧但越慢 (默认 5)",
    )
    parser.add_argument(
        "--score_smooth_alpha",
        type=float,
        default=0.5,
        help="置信度分数平滑系数 (0~1)，越小响应越快 (默认 0.5)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="推理完成后输出带人脸框的标注视频",
    )
    parser.add_argument(
        "--annotate_dir",
        type=str,
        default="./测试结果_标注视频",
        help="标注视频输出目录 (默认 ./测试结果_标注视频)",
    )
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="仅测试人脸检测，跳过模型加载和推理，自动输出标注视频",
    )
    parser.add_argument(
        "--face_detector_model",
        type=str,
        default="blaze_face_full_range.tflite",
        help="MediaPipe 人脸检测模型路径",
    )
    parser.add_argument(
        "--mediapipe_lip_crop",
        type=int,
        default=1,
        help="以嘴部关键点为中心做参考裁剪（边长按人脸框比例缩放）",
    )
    parser.add_argument(
        "--mediapipe_lip_crop_scale",
        type=float,
        default=0.55,
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
        "--mediapipe_detect_max_side",
        type=int,
        default=320,
        help="MediaPipe 检测前降采样的最大边长（像素），0=不降采样",
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
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe 人脸检测最低置信度阈值 (0~1)，低于此值的检测结果被丢弃，框颜色为黄色/红色 (默认 0.5)",
    )
    parser.add_argument(
        "--files",
        type=str,
        default="",
        help="只处理指定编号，逗号分隔，如 6,16,40,48,75,84；留空=全部处理",
    )
    args = parser.parse_args()

    file_filter = None
    if args.files:
        file_filter = set(s.strip() for s in args.files.split(",") if s.strip())
        print(f"文件过滤: 只处理 {sorted(file_filter)}")

    sdk_kwargs = dict(
        infer_chunk_ms=args.infer_chunk_ms,
        context_ms=args.context_ms,
        max_history_ms=args.max_history_ms,
        use_stream_cache=args.use_stream_cache,
        face_detector="none",
        face_detector_model_path=args.face_detector_model,
        mediapipe_use_lip_center_crop=1 if args.mediapipe_lip_crop else 0,
        mediapipe_lip_crop_scale=float(args.mediapipe_lip_crop_scale),
        mediapipe_lip_crop_min_px=int(args.mediapipe_lip_crop_min_px),
        mediapipe_lip_crop_max_px=int(args.mediapipe_lip_crop_max_px),
        face_target_policy=str(args.face_target_policy),
        face_target_lock=int(args.face_target_lock),
        face_target_lock_min_iou=float(args.face_target_lock_min_iou),
    )
    if not args.face_only:
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
    # audio_dir = "./测试用例/音频"
    # video_dir = "./测试用例/视频"
    audio_dir = "./测试用例/测试用例/audio_wav"
    video_dir = "./测试用例/测试用例/video"

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
        if file_filter and base_name not in file_filter:
            continue
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

        # === Part 1: 输入层 — 按 chunk_ms 切分整段文件，模拟流式输入 ===
        chunk_ms = float(args.chunk_ms)
        v_per_chunk = max(1, int(round(fps * chunk_ms / 1000.0)))

        if args.face_only:
            # --- 人脸检测测试模式：跳过所有推理 ---
            preprocessor = VisualPreprocessor(
                crop_size=128,
                model_path=str(args.face_detector_model),
                face_scale=float(args.face_scale),
                detect_every_n=int(args.detect_every_n),
                detect_max_side=int(args.mediapipe_detect_max_side),
                score_smooth_alpha=float(args.score_smooth_alpha),
                min_detection_confidence=float(args.min_detection_confidence),
                use_lip_center_crop=bool(int(args.mediapipe_lip_crop)),
                lip_crop_scale=float(args.mediapipe_lip_crop_scale),
                lip_crop_min_px=int(args.mediapipe_lip_crop_min_px),
                lip_crop_max_px=int(args.mediapipe_lip_crop_max_px),
                target_policy=str(args.face_target_policy),
                target_lock=bool(int(args.face_target_lock)),
                target_lock_min_iou=float(args.face_target_lock_min_iou),
            )
            num_chunks = (len(frames) + v_per_chunk - 1) // v_per_chunk
            all_face_boxes: list = []
            all_face_box_colors: list = []
            for ci in range(num_chunks):
                v_start = ci * v_per_chunk
                v_frames = frames[v_start : v_start + v_per_chunk]
                if not v_frames:
                    break
                r = preprocessor.process_chunk(v_frames)
                all_face_boxes.extend(r["face_boxes"])
                all_face_box_colors.extend(r["face_box_colors"])

            if all_face_boxes and any(b is not None for b in all_face_boxes):
                annotate_dir = str(args.annotate_dir)
                os.makedirs(annotate_dir, exist_ok=True)
                out_vid = os.path.join(annotate_dir, f"{base_name}_annotated.mp4")
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))
                n = min(len(all_face_boxes), len(frames))
                for i in range(n):
                    frame = frames[i].copy()
                    box = all_face_boxes[i]
                    color = all_face_box_colors[i]
                    if box is not None and color is not None:
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                    writer.write(frame)
                writer.release()
                print(f"已保存标注视频: {out_vid}")
            else:
                print(f"{base_name}: 未检测到人脸，无标注视频")
            continue

        audio_step = max(1, int(round(sr * chunk_ms / 1000.0)))
        total_samples = wav.shape[1]
        num_chunks = (total_samples + audio_step - 1) // audio_step

        # === Part 2: 视觉处理层 — MediaPipe 检测 ===
        crop_size = int(streamer._core._tracker_args["crop_size"])
        preprocessor = VisualPreprocessor(
            crop_size=crop_size,
            model_path=str(args.face_detector_model),
            detect_every_n=int(args.detect_every_n),
            detect_max_side=int(args.mediapipe_detect_max_side),
            face_scale=float(args.face_scale),
            min_detection_confidence=float(args.min_detection_confidence),
            use_lip_center_crop=bool(int(args.mediapipe_lip_crop)),
            lip_crop_scale=float(args.mediapipe_lip_crop_scale),
            lip_crop_min_px=int(args.mediapipe_lip_crop_min_px),
            lip_crop_max_px=int(args.mediapipe_lip_crop_max_px),
            target_policy=str(args.face_target_policy),
            target_lock=bool(int(args.face_target_lock)),
            target_lock_min_iou=float(args.face_target_lock_min_iou),
        )

        # === Part 2+3: StreamProcessor ===
        processor = StreamProcessor(
            streamer=streamer,
            preprocessor=preprocessor,
            sr=sr,
            fps=fps,
            infer_chunk_ms=float(args.infer_chunk_ms),
            record_crops=bool(args.write_lip),
        )

        outputs_all: list = []

        for ci in range(num_chunks):
            a_start = ci * audio_step
            a_end = min(a_start + audio_step, total_samples)
            a_chunk = wav[:, a_start:a_end]

            v_start = ci * v_per_chunk
            v_frames = frames[v_start : v_start + v_per_chunk]
            if not v_frames:
                break

            segs = processor.feed_chunk(a_chunk, v_frames)
            outputs_all.extend(segs)

        # Tail flush：残留数据（不足 infer_chunk_ms）
        segs = processor.flush()
        outputs_all.extend(segs)

        # --- 写裁剪视频 ---
        if args.write_lip:
            recorded = processor.get_recorded_crops()
            if recorded and recorded[0]:
                lip_dir = str(args.write_lip_dir)
                os.makedirs(lip_dir, exist_ok=True)
                out_lip = os.path.join(lip_dir, f"{base_name}_lip.mp4")
                h, w = recorded[0][0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_lip, fourcc, fps, (w, h))
                for crops in recorded:
                    for crop in crops:
                        frame_bgr = np.clip(crop, 0, 255).astype(np.uint8)[:, :, ::-1]
                        writer.write(frame_bgr)
                writer.release()
                print(f"已保存嘴唇视频: {out_lip}")
            else:
                print(f"{base_name}: 未检测到人脸，无嘴唇视频")

        # --- 写标注视频（测试用，离线渲染） ---
        if args.annotate:
            all_boxes = processor.get_face_boxes()
            all_colors = processor.get_face_box_colors()
            if all_boxes and any(b is not None for b in all_boxes):
                annotate_dir = str(args.annotate_dir)
                os.makedirs(annotate_dir, exist_ok=True)
                out_vid = os.path.join(annotate_dir, f"{base_name}_annotated.mp4")
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))
                n = min(len(all_boxes), len(frames))
                for i in range(n):
                    frame = frames[i].copy()
                    box = all_boxes[i]
                    color = all_colors[i]
                    if box is not None and color is not None:
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                    writer.write(frame)
                writer.release()
                print(f"已保存标注视频: {out_vid}")
            else:
                print(f"{base_name}: 未检测到人脸，无标注视频")

        # RTF
        audio_dur_s = processor.total_infer_audio_duration_s
        face_s = processor.sum_face_detection_s
        infer_s = processor.sum_inference_s
        total_s = processor.sum_total_s
        rtf = total_s / audio_dur_s if audio_dur_s > 1e-9 else float("nan")
        print(f"RTF: {rtf:.3f}  (face={face_s:.3f}s + infer={infer_s:.3f}s = {total_s:.3f}s, audio={audio_dur_s:.3f}s)")

        if outputs_all:
            out_audio = np.concatenate(outputs_all, axis=0).astype(np.float32, copy=False)
            out_path = os.path.join(out_dir, f"{base_name}_out.wav")
            sf.write(out_path, out_audio, int(sr), subtype="PCM_16")
            print(f"已保存: {out_path}")
        else:
            print(f"{base_name}: 无输出")

    if not args.face_only:
        streamer.close()


if __name__ == "__main__":
    main()
