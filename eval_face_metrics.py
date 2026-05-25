#!/usr/bin/env python3
"""Face tracking metrics: IoU (detection) and simplified MOTA (single-speaker)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from av_stream_inference import FaceHaarStreamTracker, _draw_overlay_box_with_label, _overlay_font_params


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


def load_video_frames_bgr(mp4_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {mp4_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 25.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {mp4_path}")
    return frames, fps


def create_tracker(
    face_detector: str,
    model_path: str,
    detect_every_n: int,
    face_scale: float,
    box_smooth_alpha: float,
    mediapipe_use_lip_center_crop: bool = False,
    mediapipe_lip_crop_scale: float = 0.55,
    mediapipe_lip_crop_min_px: int = 48,
    mediapipe_lip_crop_max_px: int = 2048,
    target_policy: str = "center_largest",
    target_lock: bool = True,
    target_lock_min_iou: float = 0.15,
):
    if face_detector == "mediapipe":
        from face_mediapipe_tracker import FaceMediaPipeStreamTracker

        return FaceMediaPipeStreamTracker(
            detect_every_n=detect_every_n,
            face_scale=face_scale,
            box_smooth_alpha=box_smooth_alpha,
            model_path=model_path,
            use_lip_center_crop=bool(mediapipe_use_lip_center_crop),
            lip_crop_scale=float(mediapipe_lip_crop_scale),
            lip_crop_min_px=int(mediapipe_lip_crop_min_px),
            lip_crop_max_px=int(mediapipe_lip_crop_max_px),
            target_policy=str(target_policy),
            target_lock=bool(target_lock),
            target_lock_min_iou=float(target_lock_min_iou),
        )
    return FaceHaarStreamTracker(
        detect_every_n=detect_every_n,
        face_scale=face_scale,
        box_smooth_alpha=box_smooth_alpha,
        target_policy=str(target_policy),
        target_lock=bool(target_lock),
        target_lock_min_iou=float(target_lock_min_iou),
    )


def load_gt_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gt_box_for_frame(gt: Dict[str, Any], frame_idx: int) -> Optional[np.ndarray]:
    target_id = int(gt.get("target_track_id", 1))
    frame_map = {int(fr["frame_idx"]): fr for fr in gt.get("frames", [])}
    fr = frame_map.get(int(frame_idx))
    if fr is None:
        return None
    for obj in fr.get("objects", []):
        if int(obj.get("track_id", -1)) == target_id:
            return np.array(
                [obj["x1"], obj["y1"], obj["x2"], obj["y2"]], dtype=np.float32
            )
    return None


def pred_box_from_tracker(tracker) -> Optional[np.ndarray]:
    if tracker.last_box is None:
        return None
    return np.array(tracker.last_box, dtype=np.float32)


def is_fallback(tracker) -> bool:
    return tracker.last_detected_box is None


@dataclass
class VideoMetrics:
    video_id: str
    n_frames: int = 0
    n_gt: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0
    iou_below_thr_frames: int = 0
    fn: int = 0
    fp: int = 0
    idsw: int = 0
    scene_switches: int = 0


@dataclass
class EvalResult:
    per_video: Dict[str, VideoMetrics] = field(default_factory=dict)

    def summary(self, iou_thr: float) -> Dict[str, float]:
        if not self.per_video:
            return {}
        mean_iou = np.mean(
            [vm.iou_sum / max(1, vm.iou_count) for vm in self.per_video.values() if vm.iou_count > 0]
            or [0.0]
        )
        below = sum(vm.iou_below_thr_frames for vm in self.per_video.values())
        n_frames = sum(vm.n_frames for vm in self.per_video.values())
        fn = sum(vm.fn for vm in self.per_video.values())
        fp = sum(vm.fp for vm in self.per_video.values())
        idsw = sum(vm.idsw for vm in self.per_video.values())
        n_gt = sum(vm.n_gt for vm in self.per_video.values())
        mota = 1.0 - (fn + fp + idsw) / max(1, n_gt)
        return {
            "mean_iou": float(mean_iou),
            "iou_below_thr_ratio": float(below / max(1, n_frames)),
            "iou_thr": float(iou_thr),
            "mota": float(mota),
            "fn": int(fn),
            "fp": int(fp),
            "idsw": int(idsw),
            "n_gt": int(n_gt),
            "n_frames": int(n_frames),
        }


def eval_video(
    video_path: Path,
    gt_path: Optional[Path],
    tracker,
    iou_thr: float,
    scene_switch_iou_thr: float,
    debug_writer: Optional[cv2.VideoWriter] = None,
) -> VideoMetrics:
    video_id = video_path.stem
    frames, _fps = load_video_frames_bgr(str(video_path))
    gt = load_gt_json(gt_path) if gt_path and gt_path.is_file() else None
    vm = VideoMetrics(video_id=video_id, n_frames=len(frames))
    prev_gt_box: Optional[np.ndarray] = None

    for frame_idx, frame_bgr in enumerate(frames):
        _, scene_switched = tracker.process_bgr(frame_bgr, scene_switch_iou_thr=scene_switch_iou_thr)
        if scene_switched:
            vm.scene_switches += 1
            vm.idsw += 1

        pred = pred_box_from_tracker(tracker)
        fallback = is_fallback(tracker)
        gt_box = gt_box_for_frame(gt, frame_idx) if gt else None

        if debug_writer is not None and pred is not None:
            vis = frame_bgr.copy()
            hh, ww = vis.shape[:2]
            fs, bt, tt = _overlay_font_params(ww, hh)
            ib = getattr(tracker, "interferer_boxes", []) or []
            isc = getattr(tracker, "interferer_box_scores", None)
            if isc is None:
                isc = [None] * len(ib)
            else:
                isc = list(isc)
                while len(isc) < len(ib):
                    isc.append(None)
            for box, sc in zip(ib, isc):
                _draw_overlay_box_with_label(
                    vis, list(box), (0, 255, 255), sc, fs, bt, tt
                )
            ts = getattr(tracker, "target_score", None)
            _draw_overlay_box_with_label(
                vis, pred.tolist(), (0, 255, 0), ts, fs, bt, tt
            )
            debug_writer.write(vis)

        if gt_box is not None:
            vm.n_gt += 1
            if pred is None or fallback:
                vm.fn += 1
                vm.iou_below_thr_frames += 1
            else:
                iou = _box_iou_xyxy(pred, gt_box)
                vm.iou_sum += iou
                vm.iou_count += 1
                if iou < iou_thr:
                    vm.iou_below_thr_frames += 1
            prev_gt_box = gt_box.copy()
        else:
            if pred is not None and not fallback:
                vm.fp += 1

    return vm


def load_manual_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def print_manual_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n=== Manual scores ===")
    total_err = 0.0
    total_dur = 0.0
    total_sw = 0
    for row in rows:
        vid = row.get("video_id", "")
        err = float(row.get("error_detect_sec", 0) or 0)
        dur = float(row.get("total_sec", 0) or 0)
        sw = int(float(row.get("id_switch_count", 0) or 0))
        ratio = err / dur if dur > 1e-9 else float("nan")
        print(f"  {vid}: error_ratio={ratio:.4f} ({err:.2f}/{dur:.2f}s) id_switch={sw}")
        total_err += err
        total_dur += dur
        total_sw += sw
    if total_dur > 1e-9:
        print(f"  ALL: error_ratio={total_err / total_dur:.4f} id_switch_total={total_sw}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Face IoU / MOTA evaluation")
    parser.add_argument("--video-dir", type=str, default="./测试用例/视频")
    parser.add_argument("--gt-dir", type=str, default="./测试用例/face_gt")
    parser.add_argument("--face-detector", choices=["haar", "mediapipe"], default="haar")
    parser.add_argument("--face-detector-model", type=str, default="detector.tflite")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--scene-switch-iou-thr", type=float, default=0.15)
    parser.add_argument("--detect-every-n", type=int, default=5)
    parser.add_argument("--face-scale", type=float, default=1.25)
    parser.add_argument("--box-smooth-alpha", type=float, default=0.85)
    parser.add_argument(
        "--mediapipe-lip-crop",
        action="store_true",
        help="mediapipe 时使用嘴中心锚定裁剪（与推理 --mediapipe_lip_crop 对齐）",
    )
    parser.add_argument("--mediapipe-lip-crop-scale", type=float, default=0.55)
    parser.add_argument("--mediapipe-lip-crop-min-px", type=int, default=48)
    parser.add_argument("--mediapipe-lip-crop-max-px", type=int, default=2048)
    parser.add_argument(
        "--face-target-policy",
        choices=["largest", "center", "center_largest", "center_largest_lock"],
        default="center_largest",
        help="multi-face target: center_largest picks face near image center among similar sizes",
    )
    parser.add_argument("--face-target-lock", type=int, choices=[0, 1], default=1)
    parser.add_argument("--face-target-lock-min-iou", type=float, default=0.15)
    parser.add_argument("--manual-csv", type=str, default="")
    parser.add_argument("--debug-video-dir", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--single", type=str, default="", help="Only evaluate video id, e.g. 03")
    args = parser.parse_args()

    if args.manual_csv:
        print_manual_summary(load_manual_csv(Path(args.manual_csv)))

    video_dir = Path(args.video_dir)
    gt_dir = Path(args.gt_dir)
    debug_dir = Path(args.debug_video_dir) if args.debug_video_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if args.single:
        videos = [video_dir / f"{args.single}.mp4"]

    result = EvalResult()
    for vp in videos:
        if not vp.is_file():
            print(f"skip missing: {vp}")
            continue
        gt_path = gt_dir / f"{vp.stem}.json"
        if not gt_path.is_file():
            print(f"skip {vp.name}: no GT {gt_path}")
            continue

        tracker = create_tracker(
            args.face_detector,
            args.face_detector_model,
            args.detect_every_n,
            args.face_scale,
            args.box_smooth_alpha,
            mediapipe_use_lip_center_crop=bool(args.mediapipe_lip_crop),
            mediapipe_lip_crop_scale=float(args.mediapipe_lip_crop_scale),
            mediapipe_lip_crop_min_px=int(args.mediapipe_lip_crop_min_px),
            mediapipe_lip_crop_max_px=int(args.mediapipe_lip_crop_max_px),
            target_policy=str(args.face_target_policy),
            target_lock=bool(int(args.face_target_lock)),
            target_lock_min_iou=float(args.face_target_lock_min_iou),
        )
        debug_writer = None
        if debug_dir:
            frames, fps = load_video_frames_bgr(str(vp))
            h, w = frames[0].shape[:2]
            out_mp4 = debug_dir / f"{vp.stem}_face_debug.mp4"
            debug_writer = cv2.VideoWriter(
                str(out_mp4),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        vm = eval_video(
            vp,
            gt_path,
            tracker,
            iou_thr=float(args.iou_thr),
            scene_switch_iou_thr=float(args.scene_switch_iou_thr),
            debug_writer=debug_writer,
        )
        if debug_writer is not None:
            debug_writer.release()
            print(f"debug video: {debug_dir / (vp.stem + '_face_debug.mp4')}")

        result.per_video[vp.stem] = vm
        miou = vm.iou_sum / max(1, vm.iou_count)
        mota = 1.0 - (vm.fn + vm.fp + vm.idsw) / max(1, vm.n_gt)
        print(
            f"{vp.stem}: mean_iou={miou:.3f} low_iou_ratio={vm.iou_below_thr_frames / max(1, vm.n_frames):.3f} "
            f"mota={mota:.3f} fn={vm.fn} fp={vm.fp} idsw={vm.idsw} n_gt={vm.n_gt}"
        )

    if result.per_video:
        summ = result.summary(float(args.iou_thr))
        print("\n=== Summary ===")
        for k, v in summ.items():
            print(f"  {k}: {v}")
        if args.out_json:
            out = {
                "summary": summ,
                "per_video": {
                    vid: {
                        "mean_iou": vm.iou_sum / max(1, vm.iou_count),
                        "iou_below_thr_ratio": vm.iou_below_thr_frames / max(1, vm.n_frames),
                        "mota": 1.0 - (vm.fn + vm.fp + vm.idsw) / max(1, vm.n_gt),
                        "fn": vm.fn,
                        "fp": vm.fp,
                        "idsw": vm.idsw,
                        "n_gt": vm.n_gt,
                    }
                    for vid, vm in result.per_video.items()
                },
            }
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
