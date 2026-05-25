from __future__ import annotations

from typing import List, Sequence, Optional, Tuple

import numpy as np

from av_stream_inference import AVStreamInference


class StreamInferenceSDK:
    """
    SDK 封装（对业务侧暴露 numpy 输入接口）：
    - 音频：np.ndarray，形状 (T,) 或 (C,T)，默认按 axis=0 mean 混单通道
    - 视频：frames 列表，List[np.ndarray(H,W,3)]，BGR/uint8
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
        onnx_num_threads: int = 8,
        ts_path: Optional[str] = None,
        save_face_video_path: Optional[str] = None,
        overlay_write_stride: int = 1,
        overlay_scale: float = 1.0,
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
        resolved_use_stream_cache = int(use_stream_cache)
        self._core = AVStreamInference(
            config=config,
            checkpoint_dir=checkpoint_dir,
            use_cuda_override=int(use_cuda_override),
            cpu_threads=int(cpu_threads),
            use_stream_cache=int(resolved_use_stream_cache),
            context_ms=float(context_ms),
            infer_chunk_ms=float(infer_chunk_ms),
            max_history_ms=float(max_history_ms),
            onnx_path=onnx_path,
            onnx_num_threads=int(onnx_num_threads),
            ts_path=ts_path,
            save_face_video_path=save_face_video_path,
            overlay_write_stride=int(overlay_write_stride),
            overlay_scale=float(overlay_scale),
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
        # FIFO 缓存：上游可能给 <200ms 或 >200ms 的对齐 chunk，这里统一缓存后按 200ms 切块推理
        self._audio_buf = np.array([], dtype=np.float32)  # [T]
        self._video_buf: List[np.ndarray] = []  # List[frame]
        self._need_core_start = True  # 下一个实际推理块要带 is_start=True

    @property
    def audio_sr(self) -> int:
        return int(self._core.audio_sr)

    @staticmethod
    def _audio_to_mono_numpy(wav: np.ndarray) -> np.ndarray:
        """按 `wav.mean(dim=0).numpy()` 的意图：numpy 输入统一混单通道，输出 (T,) float32。"""
        arr = np.asarray(wav)
        if arr.ndim == 2:
            arr = arr.mean(axis=0, dtype=np.float32)
        elif arr.ndim != 1:
            raise ValueError(f"wav must be 1D or 2D [C,T], got shape={arr.shape}")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _validate_frames(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        fs = list(frames)
        if not fs:
            raise ValueError("frames is empty")
        out: List[np.ndarray] = []
        for f in fs:
            frm = np.asarray(f)
            if frm.ndim != 3 or int(frm.shape[2]) != 3:
                raise ValueError(f"frame must be [H,W,3], got shape={frm.shape}")
            if frm.dtype != np.uint8:
                frm = np.clip(frm, 0, 255).astype(np.uint8)
            out.append(frm)
        return out

    def stream_inference(
        self,
        audio_chunk: np.ndarray,
        video_chunk: List[np.ndarray],
        is_start: bool = False,
        is_end: bool = False,
        sampling_rate: int = 16000,
        fps: float = 25.0,
    ) -> List[np.ndarray]:
        audio = self._audio_to_mono_numpy(audio_chunk)
        video = self._validate_frames(video_chunk)
        fps_f = float(fps) if np.isfinite(float(fps)) and float(fps) > 1e-3 else float(self.default_fps)
        return self._core.stream_inference(
            audio_chunk=audio,
            video_chunk=video,
            is_start=bool(is_start),
            is_end=bool(is_end),
            sampling_rate=int(sampling_rate),
            video_fps=fps_f,
        )

    def process_av_stream(
        self,
        audio_chunk: np.ndarray,
        video_chunk: Sequence[np.ndarray],
        is_start: bool = False,
        is_end: bool = False,
        sampling_rate: int = 16000,
        fps: float = 25.0,
    ) -> List[np.ndarray]:
        """
        单次喂入（上游已对齐的）音频/视频 chunk。

        规则：
        - 缓存用 FIFO 队列
        - 固定 200ms 为一个推理块：16kHz -> 3200 samples；25fps -> 5 frames
        - 输入 <200ms：先缓存，凑够 200ms 再推
        - 输入 >200ms：缓存后自动切成多个 200ms 推理块
        - is_end=True：如果还剩不足 200ms 的尾巴，允许尾巴推一次
        """
        sr = int(sampling_rate)
        if sr <= 0:
            raise ValueError(f"invalid sampling_rate={sampling_rate}")
        fps_f = float(fps) if np.isfinite(float(fps)) and float(fps) > 1e-3 else float(self.default_fps)

        if bool(is_start):
            self._audio_buf = np.array([], dtype=np.float32)
            self._video_buf = []
            self._need_core_start = True

        a_in = self._audio_to_mono_numpy(audio_chunk)
        v_in = self._validate_frames(video_chunk)

        # 入队（FIFO）
        if a_in.size > 0:
            self._audio_buf = (
                np.concatenate([self._audio_buf, a_in.astype(np.float32, copy=False)])
                if self._audio_buf.size
                else a_in.astype(np.float32, copy=False)
            )
        if v_in:
            self._video_buf.extend(v_in)

        # 按 infer_chunk_ms 出包
        need_audio = int(round(float(sr) * (self._infer_chunk_ms / 1000.0)))
        need_video = int(round(float(fps_f) * (self._infer_chunk_ms / 1000.0)))
        need_audio = max(1, need_audio)
        need_video = max(1, need_video)

        outputs_all: List[np.ndarray] = []

        def _pop_one_block() -> Tuple[np.ndarray, List[np.ndarray]]:
            a = self._audio_buf[:need_audio]
            self._audio_buf = self._audio_buf[need_audio:]
            v = self._video_buf[:need_video]
            del self._video_buf[:need_video]
            return a, v

        while int(self._audio_buf.shape[0]) >= need_audio and len(self._video_buf) >= need_video:
            a_blk, v_blk = _pop_one_block()
            segs = self.stream_inference(
                audio_chunk=a_blk,
                video_chunk=v_blk,
                is_start=bool(self._need_core_start),
                is_end=False,
                sampling_rate=sr,
                fps=fps_f,
            )
            self._need_core_start = False
            outputs_all.extend(segs)

        # 尾包：允许不足 200ms 推一次（要求两路都有数据）
        if bool(is_end):
            if int(self._audio_buf.shape[0]) > 0 and len(self._video_buf) > 0:
                a_tail = self._audio_buf
                v_tail = self._video_buf
                self._audio_buf = np.array([], dtype=np.float32)
                self._video_buf = []
                segs = self.stream_inference(
                    audio_chunk=a_tail,
                    video_chunk=v_tail,
                    is_start=bool(self._need_core_start),
                    is_end=True,
                    sampling_rate=sr,
                    fps=fps_f,
                )
                self._need_core_start = True
                outputs_all.extend(segs)
            else:
                # 没有尾巴可推，仍然重置状态（保证下一路流干净）
                self._audio_buf = np.array([], dtype=np.float32)
                self._video_buf = []
                self._need_core_start = True

        return outputs_all

    def close(self) -> None:
        self._core.close()

