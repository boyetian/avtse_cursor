"""独立视觉预处理模块：MediaPipe 人脸检测 + 嘴部裁剪。

与推理引擎零耦合，可独立部署为 Phase 1。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from face_mediapipe_tracker import FaceMediaPipeStreamTracker


class VisualPreprocessor:
    def __init__(
        self,
        crop_size: int = 128,
        face_scale: float = 1.25,
        detect_every_n: int = 5,
        box_smooth_alpha: float = 0.85,
        model_path: str = "detector.tflite",
        min_detection_confidence: float = 0.5,
        use_lip_center_crop: bool = True,
        lip_crop_scale: float = 0.55,
        lip_crop_min_px: int = 48,
        lip_crop_max_px: int = 2048,
        target_policy: str = "center_largest",
        target_lock: bool = True,
        target_lock_min_iou: float = 0.15,
    ):
        self._tracker = FaceMediaPipeStreamTracker(
            crop_size=int(crop_size),
            face_scale=float(face_scale),
            detect_every_n=int(detect_every_n),
            box_smooth_alpha=float(box_smooth_alpha),
            model_path=str(model_path),
            min_detection_confidence=float(min_detection_confidence),
            use_lip_center_crop=bool(use_lip_center_crop),
            lip_crop_scale=float(lip_crop_scale),
            lip_crop_min_px=int(lip_crop_min_px),
            lip_crop_max_px=int(lip_crop_max_px),
            target_policy=str(target_policy),
            target_lock=bool(target_lock),
            target_lock_min_iou=float(target_lock_min_iou),
        )
        self._any_face_detected = False

    @property
    def any_face_detected(self) -> bool:
        return self._any_face_detected

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        crop, _scene_switched = self._tracker.process_bgr(frame_bgr)
        face_detected = self._tracker.last_box is not None
        if face_detected:
            self._any_face_detected = True

        mouth_center = None
        if self._tracker.last_lip_xy is not None:
            mouth_center = (float(self._tracker.last_lip_xy[0]), float(self._tracker.last_lip_xy[1]))

        return {
            "crop": crop,
            "face_detected": face_detected,
            "mouth_center": mouth_center,
            "face_box": list(self._tracker.last_box) if self._tracker.last_box is not None else None,
            "detection_score": self._tracker.target_score,
        }

    def process_all_frames(self, frames: list) -> dict:
        self._any_face_detected = False
        crops: List[np.ndarray] = []
        face_valid: List[bool] = []
        metadata: List[dict] = []

        for frame_bgr in frames:
            result = self.process_frame(frame_bgr)
            crops.append(result["crop"])
            face_valid.append(result["face_detected"])
            metadata.append(
                {
                    "mouth_center": result["mouth_center"],
                    "face_box": result["face_box"],
                    "detection_score": result["detection_score"],
                }
            )

        return {"crops": crops, "face_valid": face_valid, "metadata": metadata}

    def process_chunk(self, frames: list) -> dict:
        """处理一个 chunk 的视频帧，tracker 状态跨 chunk 保持。
        返回 {'crops': list, 'face_valid': list, 'any_detected': bool}"""
        crops: List[np.ndarray] = []
        face_valid: List[bool] = []
        chunk_has_face = False

        for frame_bgr in frames:
            result = self.process_frame(frame_bgr)
            crops.append(result["crop"])
            face_valid.append(result["face_detected"])
            if result["face_detected"]:
                chunk_has_face = True

        return {"crops": crops, "face_valid": face_valid, "any_detected": chunk_has_face}
