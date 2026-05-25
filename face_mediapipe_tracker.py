"""流式人脸框跟踪：MediaPipe FaceDetector + 平滑（与 FaceHaarStreamTracker 同接口）。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from av_stream_inference import FaceHaarStreamTracker, pick_target_detection


def _mediapipe_detection_score(det) -> Optional[float]:
    if det.categories and len(det.categories) > 0:
        return float(det.categories[0].score)
    return None


class FaceMediaPipeStreamTracker:
    def __init__(
        self,
        crop_size: int = 128,
        face_scale: float = 1.25,
        detect_every_n: int = 5,
        box_smooth_alpha: float = 0.85,
        model_path: str = "detector.tflite",
        min_detection_confidence: float = 0.5,
        use_lip_center_crop: bool = False,
        mouth_keypoint_indices: Tuple[int, ...] = (3,),
        lip_crop_scale: float = 0.55,
        lip_crop_min_px: int = 48,
        lip_crop_max_px: int = 2048,
        target_policy: str = "center_largest",
        target_lock: bool = True,
        target_lock_min_iou: float = 0.15,
    ):
        self.target_policy = str(target_policy).lower().strip()
        self.target_lock = bool(target_lock) or self.target_policy.endswith("_lock")
        self.target_lock_min_iou = float(target_lock_min_iou)
        self.crop_size = int(crop_size)
        self.face_scale = float(face_scale)
        self.detect_every_n = max(1, int(detect_every_n))
        self.box_smooth_alpha = float(np.clip(box_smooth_alpha, 0.0, 1.0))
        self.use_lip_center_crop = bool(use_lip_center_crop)
        self._mouth_keypoint_indices = tuple(int(i) for i in mouth_keypoint_indices)
        self.lip_crop_scale = float(lip_crop_scale)
        self.lip_crop_min_px = max(2, int(lip_crop_min_px))
        self.lip_crop_max_px = max(self.lip_crop_min_px, int(lip_crop_max_px))
        self.last_box = None
        self.frame_idx = 0
        self.last_detected_box = None
        self.last_lip_xy: Optional[np.ndarray] = None
        self.interferer_boxes: List = []
        self.interferer_box_scores: List[Optional[float]] = []
        self.target_score: Optional[float] = None

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=float(min_detection_confidence),
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    @staticmethod
    def _bbox_to_square_xyxy(x1: int, y1: int, x2: int, y2: int, face_scale: float, w: int, h: int) -> np.ndarray:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s = max(x2 - x1, y2 - y1) * face_scale
        box = np.array([cx - s / 2.0, cy - s / 2.0, cx + s / 2.0, cy + s / 2.0], dtype=np.float32)
        box[0] = np.clip(box[0], 0, w - 1)
        box[1] = np.clip(box[1], 0, h - 1)
        box[2] = np.clip(box[2], box[0] + 1, w)
        box[3] = np.clip(box[3], box[1] + 1, h)
        return box

    def _lip_center_xy_from_detection(self, det, w: int, h: int) -> Optional[np.ndarray]:
        """BlazeFace 约定：0 右眼 1 左眼 2 鼻尖 3 嘴中心 4/5 耳；mouth_keypoint_indices 默认可为 (3,)。"""
        kps = getattr(det, "keypoints", None)
        if not kps:
            return None
        xs: List[float] = []
        ys: List[float] = []
        for i in self._mouth_keypoint_indices:
            if i < 0 or i >= len(kps):
                return None
            kp = kps[i]
            xs.append(float(kp.x) * float(w))
            ys.append(float(kp.y) * float(h))
        return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)

    @staticmethod
    def _crop_square_centered(frame_bgr: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
        """以 (cx, cy) 为中心取 size×size 正方形；越界部分黑边填充。"""
        h, w = frame_bgr.shape[:2]
        half = int(size) // 2
        x1 = int(np.floor(float(cx) - float(half)))
        y1 = int(np.floor(float(cy) - float(half)))
        x2 = x1 + int(size)
        y2 = y1 + int(size)
        canvas = np.zeros((int(size), int(size), 3), dtype=np.uint8)
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return canvas
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame_bgr[src_y1:src_y2, src_x1:src_x2]
        return canvas

    def _detect_all_boxes(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float], List[Tuple[List[float], Optional[float]]], Optional[np.ndarray]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.detections:
            return None, None, [], None
        h, w = frame_bgr.shape[:2]
        boxes: List[Tuple[float, np.ndarray, Optional[float], Optional[np.ndarray]]] = []
        for det in result.detections:
            bb = det.bounding_box
            x1 = int(bb.origin_x)
            y1 = int(bb.origin_y)
            x2 = int(bb.origin_x + bb.width)
            y2 = int(bb.origin_y + bb.height)
            sq = self._bbox_to_square_xyxy(x1, y1, x2, y2, self.face_scale, w, h)
            area = float(max(0.0, sq[2] - sq[0]) * max(0.0, sq[3] - sq[1]))
            sc = _mediapipe_detection_score(det)
            lip = self._lip_center_xy_from_detection(det, w, h)
            boxes.append((area, sq, sc, lip))
        ti = pick_target_detection(
            boxes,
            w,
            h,
            policy=self.target_policy,
            locked_box=self._locked_box_for_pick(),
            lock_min_iou=self.target_lock_min_iou,
        )
        if ti < 0:
            interferer = [(boxes[i][1].tolist(), boxes[i][2]) for i in range(len(boxes))]
            return None, None, interferer, None
        target = boxes[ti][1]
        target_score = boxes[ti][2]
        target_lip = boxes[ti][3]
        interferer = [(boxes[i][1].tolist(), boxes[i][2]) for i in range(len(boxes)) if i != ti]
        return target, target_score, interferer, target_lip

    def _locked_box_for_pick(self) -> Optional[np.ndarray]:
        if not self.target_lock:
            return None
        return self.last_detected_box

    def process_bgr(self, frame_bgr: np.ndarray, scene_switch_iou_thr: float = 0.15) -> Tuple[np.ndarray, bool]:
        h, w = frame_bgr.shape[:2]
        run_det = (self.frame_idx % self.detect_every_n == 0) or (self.last_box is None)
        scene_switched = False
        self.interferer_boxes = []
        self.interferer_box_scores = []

        if run_det:
            new_box, det_target_score, inter_pairs, lip_obs = self._detect_all_boxes(frame_bgr)
            self.interferer_boxes = [p[0] for p in inter_pairs]
            self.interferer_box_scores = [p[1] for p in inter_pairs]
            if new_box is not None:
                if self.last_detected_box is not None and not self.target_lock:
                    iou = FaceHaarStreamTracker._box_iou_xyxy(self.last_detected_box, new_box)
                    if float(iou) < float(scene_switch_iou_thr):
                        scene_switched = True
                self.last_detected_box = new_box.copy()
                self.target_score = det_target_score
                if self.last_box is None or self.box_smooth_alpha >= 0.999:
                    self.last_box = new_box.tolist()
                else:
                    prev = np.array(self.last_box, dtype=np.float32)
                    smoothed = self.box_smooth_alpha * prev + (1.0 - self.box_smooth_alpha) * new_box
                    self.last_box = smoothed.tolist()

                if self.use_lip_center_crop:
                    if lip_obs is not None:
                        if self.last_lip_xy is None or self.box_smooth_alpha >= 0.999:
                            self.last_lip_xy = lip_obs.astype(np.float32, copy=True)
                        else:
                            self.last_lip_xy = self.box_smooth_alpha * self.last_lip_xy + (
                                1.0 - self.box_smooth_alpha
                            ) * lip_obs
                    else:
                        self.last_lip_xy = None
                else:
                    self.last_lip_xy = None
            else:
                self.last_box = None
                self.last_detected_box = None
                self.target_score = None
                self.last_lip_xy = None

        if self.last_box is None:
            self.frame_idx += 1
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32), bool(scene_switched)

        if self.use_lip_center_crop and self.last_lip_xy is not None:
            bx1, by1, bx2, by2 = [float(v) for v in self.last_box]
            face_side = max(bx2 - bx1, by2 - by1)
            side_px = int(round(float(self.lip_crop_scale) * float(face_side)))
            side_px = max(self.lip_crop_min_px, min(self.lip_crop_max_px, side_px))
            patch = self._crop_square_centered(
                frame_bgr,
                float(self.last_lip_xy[0]),
                float(self.last_lip_xy[1]),
                side_px,
            )
            if patch.shape[0] != self.crop_size or patch.shape[1] != self.crop_size:
                patch = cv2.resize(patch, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            crop = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            self.frame_idx += 1
            return crop.astype(np.float32), bool(scene_switched)

        x1, y1, x2, y2 = self.last_box
        x1 = int(round(max(0, min(w - 1, float(x1)))))
        y1 = int(round(max(0, min(h - 1, float(y1)))))
        x2 = int(round(max(x1 + 1, min(w, float(x2)))))
        y2 = int(round(max(y1 + 1, min(h, float(y2)))))

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            cs = min(h, w)
            x1 = (w - cs) // 2
            y1 = (h - cs) // 2
            crop = frame_bgr[y1 : y1 + cs, x1 : x1 + cs]
        crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        self.frame_idx += 1
        return crop.astype(np.float32), bool(scene_switched)
