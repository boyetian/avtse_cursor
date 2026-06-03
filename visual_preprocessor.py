"""视觉模块：MediaPipe 人脸检测 + 跟踪 + 裁剪 + 对外接口。

零依赖推理引擎，可独立部署。
对外 API：VisualPreprocessor, confidence_to_bbox_color
内部工具（av_stream_inference 也引用）：_box_iou_xyxy, pick_target_detection
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return (inter / den) if den > 1e-9 else 0.0


def pick_target_detection(
    boxes: Sequence[Tuple[float, np.ndarray, Optional[float], Optional[np.ndarray]]],
    frame_w: int,
    frame_h: int,
) -> int:
    """从多人脸检测候选中选目标索引，使用 center_largest 策略。

    规则：面积最大者优先；若多个面积差 < 5%，选最靠近画面中心的。
    元素为 (area, box_xyxy, score|None, lip_xy|None)。
    """
    if not boxes:
        return 0

    max_area = max(float(b[0]) for b in boxes)
    thr = 0.95 * max_area
    candidates = [i for i, b in enumerate(boxes) if float(b[0]) >= thr - 1e-6]
    if not candidates:
        candidates = list(range(len(boxes)))

    cx_img = float(frame_w) * 0.5
    cy_img = float(frame_h) * 0.5

    def _center_dist_sq(i: int) -> float:
        box = boxes[i][1]
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        return (cx - cx_img) ** 2 + (cy - cy_img) ** 2

    return int(min(candidates, key=_center_dist_sq))


def _mediapipe_detection_score(det) -> Optional[float]:
    if det.categories and len(det.categories) > 0:
        return float(det.categories[0].score)
    return None


class FaceMediaPipeStreamTracker:
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
        use_lip_center_crop: bool = False,
        mouth_keypoint_indices: Tuple[int, ...] = (3,),
        lip_crop_scale: float = 0.55,
        lip_crop_min_px: int = 48,
        lip_crop_max_px: int = 2048,
        area_switch_ratio: float = 1.2,
        area_switch_min_frames: int = 25,
        area_switch_min_confidence: float = 0.6,
    ):
        self.crop_size = int(crop_size)
        self.face_scale = float(face_scale)
        self.detect_every_n = max(1, int(detect_every_n))
        self.detect_max_side = int(detect_max_side)
        self.box_smooth_alpha = float(np.clip(box_smooth_alpha, 0.0, 1.0))
        self.score_smooth_alpha = float(np.clip(score_smooth_alpha, 0.0, 1.0))
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
        self.area_switch_ratio = float(area_switch_ratio)
        self.area_switch_min_frames = max(1, int(area_switch_min_frames))
        self.area_switch_min_confidence = float(area_switch_min_confidence)
        self._candidate_box: Optional[np.ndarray] = None
        self._candidate_frames: int = 0
        self._target_lost_frames: int = 0

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
        h, w = frame_bgr.shape[:2]
        sx = sy = 1.0
        if self.detect_max_side > 0:
            ms = max(h, w)
            if ms > self.detect_max_side:
                scale = float(self.detect_max_side) / float(ms)
                sw = max(1, int(round(w * scale)))
                sh = max(1, int(round(h * scale)))
                frame_bgr = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
                sx = w / float(sw)
                sy = h / float(sh)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.detections:
            return None, None, [], None
        boxes: List[Tuple[float, np.ndarray, Optional[float], Optional[np.ndarray]]] = []
        for det in result.detections:
            bb = det.bounding_box
            x1 = int(round(bb.origin_x * sx))
            y1 = int(round(bb.origin_y * sy))
            x2 = int(round((bb.origin_x + bb.width) * sx))
            y2 = int(round((bb.origin_y + bb.height) * sy))
            sq = self._bbox_to_square_xyxy(x1, y1, x2, y2, self.face_scale, w, h)
            area = float(max(0.0, sq[2] - sq[0]) * max(0.0, sq[3] - sq[1]))
            sc = _mediapipe_detection_score(det)
            lip = self._lip_center_xy_from_detection(det, w, h)
            boxes.append((area, sq, sc, lip))

        if self.last_box is None:
            # 初次锁定：面积最大，5%内比中心
            ti = pick_target_detection(boxes, w, h)
        else:
            # 已锁定：IoU 匹配跟踪同一人
            last_arr = np.array(self.last_box, dtype=np.float32)
            best_iou = 0.0
            ti = -1
            for i, (_, sq, _, _) in enumerate(boxes):
                iou = _box_iou_xyxy(last_arr, sq)
                if iou > best_iou:
                    best_iou = iou
                    ti = i
            if best_iou < 0.3:
                self._target_lost_frames += 1
                ti = -1
                if self._target_lost_frames >= 5:
                    # 丢失超过约1秒(5检测帧)，确认丢失后重选
                    ti = pick_target_detection(boxes, w, h)
                    self._target_lost_frames = 0
            else:
                self._target_lost_frames = 0

        # === Area-based target switching ===
        if self.area_switch_ratio > 0:
            if ti >= 0:
                t_area = boxes[ti][0]
                t_conf = boxes[ti][2]
                t_box = boxes[ti][1]
            elif self.last_box is not None and self._target_lost_frames > 0:
                # 目标丢失但仍在保持：用旧框的面积作比较基准
                t_box = np.array(self.last_box, dtype=np.float32)
                t_area = float((t_box[2] - t_box[0]) * (t_box[3] - t_box[1]))
                t_conf = None  # 丢失中无置信度
            else:
                t_area = None

            if t_area is not None:
                # Find largest interferer (all faces when target is lost)
                best_i = -1
                best_area = 0.0
                for i in range(len(boxes)):
                    if i == ti:
                        continue
                    if boxes[i][0] > best_area:
                        best_area = boxes[i][0]
                        best_i = i

                if best_i >= 0 and best_area > self.area_switch_ratio * t_area:
                    candidate_box = boxes[best_i][1]
                    cand_overlap = _box_iou_xyxy(t_box, candidate_box)
                    if cand_overlap < 0.5:
                        conf_low = (t_conf is None) or (float(t_conf) < self.area_switch_min_confidence)
                        lip_still = self._lip_motion_detect()

                        if conf_low or lip_still:
                            matched = False
                            if self._candidate_box is not None:
                                iou = _box_iou_xyxy(self._candidate_box, candidate_box)
                                if iou >= 0.15:
                                    matched = True
                                    self._candidate_frames += 1
                            if not matched:
                                self._candidate_box = candidate_box.copy()
                                self._candidate_frames = 1

                            if self._candidate_frames >= self.area_switch_min_frames:
                                ti = best_i
                                self._candidate_box = None
                                self._candidate_frames = 0
                                self._target_lost_frames = 0  # 切换恢复
                        else:
                            self._candidate_box = None
                            self._candidate_frames = 0
                    else:
                        self._candidate_box = None
                        self._candidate_frames = 0
                else:
                    self._candidate_box = None
                    self._candidate_frames = 0

        if ti < 0:
            interferer = [(boxes[i][1].tolist(), boxes[i][2]) for i in range(len(boxes))]
            if self._target_lost_frames > 0 and self.last_box is not None:
                # 目标暂时丢失但未确认，保持上次的 target（避免丢失1帧就重选）
                target = np.array(self.last_box, dtype=np.float32)
                return target, self.target_score, interferer, None
            return None, None, interferer, None
        target = boxes[ti][1]
        target_score = boxes[ti][2]
        target_lip = boxes[ti][3]
        interferer = [(boxes[i][1].tolist(), boxes[i][2]) for i in range(len(boxes)) if i != ti]
        return target, target_score, interferer, target_lip

    def _lip_motion_detect(self) -> bool:
        """Placeholder: detect whether Active_Target lips are stationary.

        Returns:
            True if lips appear not moving (condition met for area switch).
            Currently always returns False (not implemented).
        """
        return False

    def process_bgr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        h, w = frame_bgr.shape[:2]
        run_det = (self.frame_idx % self.detect_every_n == 0) or (self.last_box is None)
        self.interferer_boxes = []
        self.interferer_box_scores = []

        if run_det:
            new_box, det_target_score, inter_pairs, lip_obs = self._detect_all_boxes(frame_bgr)
            self.interferer_boxes = [p[0] for p in inter_pairs]
            self.interferer_box_scores = [p[1] for p in inter_pairs]
            if new_box is not None:
                self.last_detected_box = new_box.copy()
                if det_target_score is not None:
                    if self.target_score is None or self.score_smooth_alpha >= 0.999:
                        self.target_score = float(det_target_score)
                    else:
                        self.target_score = (self.score_smooth_alpha * self.target_score +
                                             (1.0 - self.score_smooth_alpha) * float(det_target_score))
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
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32), False

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
            return crop.astype(np.float32), False

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
        return crop.astype(np.float32), False


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
    """视觉预处理接口：封装 FaceMediaPipeStreamTracker，输出 crops / face_valid / face_boxes / face_box_colors。"""

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
        area_switch_ratio: float = 1.2,
        area_switch_min_frames: int = 25,
        area_switch_min_confidence: float = 0.6,
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
            area_switch_ratio=float(area_switch_ratio),
            area_switch_min_frames=int(area_switch_min_frames),
            area_switch_min_confidence=float(area_switch_min_confidence),
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
