"""独立视觉预处理模块：MediaPipe 人脸检测 + 嘴部裁剪。

与推理引擎零耦合，可独立部署为 Phase 1。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from face_mediapipe_tracker import FaceMediaPipeStreamTracker


def confidence_to_bbox_color(score: Optional[float], min_score: float = 0.5) -> Optional[Tuple[int, int, int]]:
    """MediaPipe 检测置信度 → BGR 框颜色。

    Args:
        score: 检测置信度 (0~1)
        min_score: 红框最低阈值，低于此值不显示颜色
    """
    if score is None:
        return None
    s = float(score)
    m = float(min_score)
    if s >= 0.8:
        return (0, 255, 0)
    elif s >= 0.7:
        return (0, 255, 255)
    elif s >= m:
        return (0, 0, 255)
    return None


class VisualPreprocessor:
    def __init__(
        self,
        crop_size: int = 128,
        face_scale: float = 1.0,
        detect_every_n: int = 5,
        detect_max_side: int = 320,
        box_smooth_alpha: float = 0.85,
        score_smooth_alpha: float = 0.5,
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
            detect_max_side=int(detect_max_side),
            box_smooth_alpha=float(box_smooth_alpha),
            score_smooth_alpha=float(score_smooth_alpha),
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
        self._min_detection_confidence = float(min_detection_confidence)
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

        Returns:
            dict with keys:
                crops: List[np.ndarray] — (H,W,3) float32 RGB
                face_valid: List[bool]
                any_detected: bool
                face_boxes: List[Optional[List[float]]] — [x1,y1,x2,y2] per frame
                face_box_colors: List[Optional[Tuple[int,int,int]]] — BGR per frame
        """
        crops: List[np.ndarray] = []
        face_valid: List[bool] = []
        face_boxes: List[Optional[List[float]]] = []
        face_box_colors: List[Optional[Tuple[int, int, int]]] = []
        chunk_has_face = False

        for frame_bgr in frames:
            result = self.process_frame(frame_bgr)
            crops.append(result["crop"])
            face_valid.append(result["face_detected"])
            face_boxes.append(result["face_box"])
            face_box_colors.append(confidence_to_bbox_color(result["detection_score"], self._min_detection_confidence))
            if result["face_detected"]:
                chunk_has_face = True

        return {
            "crops": crops,
            "face_valid": face_valid,
            "any_detected": chunk_has_face,
            "face_boxes": face_boxes,
            "face_box_colors": face_box_colors,
        }
