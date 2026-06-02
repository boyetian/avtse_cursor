from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from av_stream_inference import AVStreamInference


class StreamInferenceSDK:
    """SDK 封装：模型初始化 + 底层推理接口。

    - 音频：np.ndarray，形状 (T,) 或 (C,T)，默认按 axis=0 mean 混单通道
    - 视频：pre-cropped frames，List[np.ndarray(H,W,3)]，float32 RGB
    """

    def __init__(
        self,
        config: str = "./checkpoints/AV_Mossformer/config.yaml",
        checkpoint_dir: str = "./checkpoints/AV_Mossformer",
        use_cuda_override: int = 0,
        cpu_threads: int = 8,
        use_stream_cache: int = 1,
        max_history_ms: float = 100.0,
        infer_chunk_ms: float = 200.0,
        context_ms: float = 100.0,
        default_fps: float = 30.0,
        onnx_path: Optional[str] = None,
        ref_onnx_path: Optional[str] = None,
        sep_onnx_path: Optional[str] = None,
        onnx_num_threads: int = 8,
        ts_path: Optional[str] = None,
        face_detector: str = "haar",
        face_detector_model_path: str = "detector.tflite",
        mediapipe_use_lip_center_crop: int = 0,
        mediapipe_lip_crop_scale: float = 0.55,
        mediapipe_lip_crop_min_px: int = 48,
        mediapipe_lip_crop_max_px: int = 2048,
        face_target_policy: str = "center_largest",
        face_target_lock: int = 1,
        face_target_lock_min_iou: float = 0.15,
    ):
        self.default_fps = float(default_fps)
        self._infer_chunk_ms = float(infer_chunk_ms)
        self._core = AVStreamInference(
            config=config,
            checkpoint_dir=checkpoint_dir,
            use_cuda_override=int(use_cuda_override),
            cpu_threads=int(cpu_threads),
            use_stream_cache=int(use_stream_cache),
            context_ms=float(context_ms),
            infer_chunk_ms=float(infer_chunk_ms),
            max_history_ms=float(max_history_ms),
            onnx_path=onnx_path,
            ref_onnx_path=ref_onnx_path,
            sep_onnx_path=sep_onnx_path,
            onnx_num_threads=int(onnx_num_threads),
            ts_path=ts_path,
            face_detector=str(face_detector),
            face_detector_model_path=str(face_detector_model_path),
            mediapipe_use_lip_center_crop=int(mediapipe_use_lip_center_crop),
            mediapipe_lip_crop_scale=float(mediapipe_lip_crop_scale),
            mediapipe_lip_crop_min_px=int(mediapipe_lip_crop_min_px),
            mediapipe_lip_crop_max_px=int(mediapipe_lip_crop_max_px),
            face_target_policy=str(face_target_policy),
            face_target_lock=int(face_target_lock),
            face_target_lock_min_iou=float(face_target_lock_min_iou),
        )

    @staticmethod
    def _audio_to_mono_numpy(wav: np.ndarray) -> np.ndarray:
        arr = np.asarray(wav)
        if arr.ndim == 2:
            arr = arr.mean(axis=0, dtype=np.float32)
        elif arr.ndim != 1:
            raise ValueError(f"wav must be 1D or 2D [C,T], got shape={arr.shape}")
        return arr.astype(np.float32, copy=False)

    def stream_inference_precropped(
        self,
        audio_chunk: np.ndarray,
        precropped_video: List[np.ndarray],
        face_valid: List[bool],
        is_start: bool = False,
        is_end: bool = False,
        sampling_rate: int = 16000,
        fps: float = 25.0,
    ) -> List[np.ndarray]:
        audio = self._audio_to_mono_numpy(audio_chunk)
        fps_f = float(fps) if np.isfinite(float(fps)) and float(fps) > 1e-3 else float(self.default_fps)
        return self._core.stream_inference_precropped(
            audio_chunk=audio,
            precropped_video=precropped_video,
            face_valid=face_valid,
            is_start=bool(is_start),
            is_end=bool(is_end),
            sampling_rate=int(sampling_rate),
            video_fps=fps_f,
        )

    def close(self) -> None:
        self._core.close()


class StreamProcessor:
    """累积音频/视频 chunk，达到 infer_chunk_ms 后做人脸 go/no-go 并推理。

    不持有文件句柄，不关心数据来源。测试模式和流式模式共用。

    用法：
        streamer = StreamInferenceSDK(...)
        preprocessor = VisualPreprocessor(...)
        processor = StreamProcessor(streamer, preprocessor, sr=16000, fps=25.0, infer_chunk_ms=500)
        while streaming:
            segs = processor.feed_chunk(audio_chunk, video_frames)
        segs = processor.flush()
        streamer.close()
    """

    def __init__(
        self,
        streamer: StreamInferenceSDK,
        preprocessor,       # VisualPreprocessor
        sr: int,
        fps: float,
        infer_chunk_ms: float,
        record_crops: bool = False,
    ):
        self._streamer = streamer
        self._preprocessor = preprocessor
        self._sr = int(sr)
        self._fps = float(fps)
        self._infer_chunk_ms = float(infer_chunk_ms)
        self._record_crops = bool(record_crops)

        self._infer_audio_samples = max(1, int(round(self._sr * self._infer_chunk_ms / 1000.0)))
        self._infer_frames = max(1, int(round(self._fps * self._infer_chunk_ms / 1000.0)))

        # Accumulators
        self._acc_audio_buf: np.ndarray = np.array([], dtype=np.float32)
        self._acc_crops: list = []
        self._acc_fv: List[bool] = []
        self._acc_face_boxes: List[Optional[List[float]]] = []
        self._acc_face_box_colors: List[Optional[Tuple[int, int, int]]] = []
        self._first_face_chunk = True

        # Crop recording
        self._recorded_crops: list = []  # list of (list of np.ndarray) per inference window

        # Timing
        self._total_infer_audio_samples = 0
        self._sum_face_detection_s = 0.0
        self._sum_inference_s = 0.0

    @property
    def total_infer_audio_duration_s(self) -> float:
        return self._total_infer_audio_samples / max(1, self._sr)

    @property
    def sum_inference_s(self) -> float:
        return self._sum_inference_s

    @property
    def sum_face_detection_s(self) -> float:
        return self._sum_face_detection_s

    @property
    def sum_total_s(self) -> float:
        return self._sum_face_detection_s + self._sum_inference_s

    def reset(self) -> None:
        self._acc_audio_buf = np.array([], dtype=np.float32)
        self._acc_crops = []
        self._acc_fv = []
        self._acc_face_boxes = []
        self._acc_face_box_colors = []
        self._first_face_chunk = True
        self._recorded_crops = []
        self._total_infer_audio_samples = 0
        self._sum_face_detection_s = 0.0
        self._sum_inference_s = 0.0

    def get_recorded_crops(self) -> list:
        """返回每个推理窗口的 crops 列表。

        Returns:
            list of (list of np.ndarray): 每个元素是一个窗口的 crops，float32 RGB。"""
        return self._recorded_crops

    def get_face_boxes(self) -> List[Optional[List[float]]]:
        """返回逐帧 Active_Target 人脸框（按 feed_chunk 喂入顺序）。

        Returns:
            List where entry i is [x1, y1, x2, y2] or None.
        """
        return list(self._acc_face_boxes)

    def get_face_box_colors(self) -> List[Optional[Tuple[int, int, int]]]:
        """返回逐帧 BGR 框颜色，与 get_face_boxes() 对齐。

        Returns:
            List where entry i is (B, G, R) or None.
            None 表示无人脸或置信度 < 0.6，调用方应使用默认颜色。
        """
        return list(self._acc_face_box_colors)

    def feed_chunk(
        self,
        audio_chunk: np.ndarray,   # (C,T) 或 (T,), float32
        video_frames: list,        # list of BGR uint8 (H,W,3)
    ) -> list:
        """喂入一个 chunk 的音频+视频帧，返回推理结果列表（可能为空）。

        Returns:
            List[np.ndarray]: 分离后的音频段，每个 (T,) float32。无人脸时返回 []。
        """
        wav = np.asarray(audio_chunk)
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]
        wav = wav.astype(np.float32, copy=False)

        t_face0 = time.perf_counter()
        r = self._preprocessor.process_chunk(video_frames)
        self._sum_face_detection_s += time.perf_counter() - t_face0

        self._acc_audio_buf = (
            np.concatenate([self._acc_audio_buf, wav], axis=1)
            if self._acc_audio_buf.size
            else wav.astype(np.float32, copy=False)
        )
        self._acc_crops.extend(r["crops"])
        self._acc_fv.extend(r["face_valid"])
        self._acc_face_boxes.extend(r.get("face_boxes", []))
        self._acc_face_box_colors.extend(r.get("face_box_colors", []))

        outputs_all: list = []

        while self._acc_audio_buf.shape[1] >= self._infer_audio_samples and len(self._acc_crops) >= self._infer_frames:
            a_seg = self._acc_audio_buf[:, :self._infer_audio_samples]
            v_seg = self._acc_crops[:self._infer_frames]
            fv_seg = self._acc_fv[:self._infer_frames]

            self._acc_audio_buf = self._acc_audio_buf[:, self._infer_audio_samples:]
            self._acc_crops = self._acc_crops[self._infer_frames:]
            self._acc_fv = self._acc_fv[self._infer_frames:]

            if any(fv_seg):
                if self._record_crops:
                    self._recorded_crops.append(v_seg)
                t0 = time.perf_counter()
                segs = self._streamer.stream_inference_precropped(
                    audio_chunk=a_seg,
                    precropped_video=v_seg,
                    face_valid=fv_seg,
                    is_start=self._first_face_chunk,
                    is_end=False,
                    sampling_rate=self._sr,
                    fps=self._fps,
                )
                self._sum_inference_s += time.perf_counter() - t0
                outputs_all.extend(segs)
                self._first_face_chunk = False
                self._total_infer_audio_samples += a_seg.shape[1]

        return outputs_all

    def flush(self) -> list:
        """清空剩余 buffer，返回尾部推理结果（is_end=True）。

        Returns:
            List[np.ndarray]: 尾部音频段。无人脸时返回 []。
        """
        outputs_all: list = []

        if self._acc_audio_buf.ndim >= 2 and self._acc_audio_buf.shape[1] > 0 and self._acc_crops:
            if any(self._acc_fv):
                if self._record_crops:
                    self._recorded_crops.append(self._acc_crops)
                t0 = time.perf_counter()
                segs = self._streamer.stream_inference_precropped(
                    audio_chunk=self._acc_audio_buf,
                    precropped_video=self._acc_crops,
                    face_valid=self._acc_fv,
                    is_start=self._first_face_chunk,
                    is_end=True,
                    sampling_rate=self._sr,
                    fps=self._fps,
                )
                self._sum_inference_s += time.perf_counter() - t0
                outputs_all.extend(segs)
                self._total_infer_audio_samples += self._acc_audio_buf.shape[1]

        self._acc_audio_buf = np.array([], dtype=np.float32)
        self._acc_crops = []
        self._acc_fv = []
        self._first_face_chunk = True
        return outputs_all
