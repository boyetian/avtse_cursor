from __future__ import annotations

import os
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import librosa
import numpy as np
import torch
import yaml
import math

from networks import network_wrapper


def _onnx_dim_to_int(dim) -> Optional[int]:
    if dim is None:
        return None
    if isinstance(dim, int):
        return int(dim)
    s = str(dim).strip()
    if not s or s.isalpha():
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


class _ONNXModelWrapper:
    """ONNX Runtime 推理包装，接口与 PyTorch model 一致，供 _run_new_hops_nonoverlap 无缝替换。

    支持 numpy 直接输入（避免 torch→numpy 转换），同时兼容 torch tensor 输入。
    若 ONNX 输入为定长，自动对 mixture/ref 时间维 pad 0 或截断。
    """

    def __init__(self, onnx_path: str, num_threads: int = 8):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.fixed_t_audio: Optional[int] = None
        self.fixed_t_ref: Optional[int] = None
        for inp in self.sess.get_inputs():
            shape = list(inp.shape)
            if inp.name == "mixture" and len(shape) >= 2:
                self.fixed_t_audio = _onnx_dim_to_int(shape[1])
            elif inp.name == "ref" and len(shape) >= 2:
                self.fixed_t_ref = _onnx_dim_to_int(shape[1])
        self.has_fixed_input = (
            self.fixed_t_audio is not None and int(self.fixed_t_audio) > 0
            and self.fixed_t_ref is not None and int(self.fixed_t_ref) > 0
        )
        if self.has_fixed_input:
            print(
                f"[ONNX] fixed input shapes: mixture [*, {self.fixed_t_audio}], "
                f"ref [*, {self.fixed_t_ref}, ...]"
            )

    @staticmethod
    def _fit_time_axis(arr: np.ndarray, axis: int, target_len: int) -> np.ndarray:
        cur = int(arr.shape[axis])
        tgt = int(target_len)
        if cur == tgt:
            return arr.astype(np.float32, copy=False)
        if cur > tgt:
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(0, tgt)
            return arr[tuple(sl)].astype(np.float32, copy=False)
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, tgt - cur)
        return np.pad(arr.astype(np.float32, copy=False), pad_width, mode="constant")

    def _prepare_inputs(self, mixture_np: np.ndarray, ref_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mix = np.asarray(mixture_np, dtype=np.float32)
        ref = np.asarray(ref_np, dtype=np.float32)
        if self.has_fixed_input:
            mix = self._fit_time_axis(mix, 1, int(self.fixed_t_audio))
            ref = self._fit_time_axis(ref, 1, int(self.fixed_t_ref))

        # RKNN-friendly ONNX: ref is 4D grayscale, need to convert from 5D RGB
        if self.normalize_mode == "mossformer" and ref.ndim == 5 and ref.shape[-1] == 3:
            # Convert RGB to grayscale using same formula as the model
            # ref shape: [B, T, H, W, 3] -> [B, T, H, W]
            ref = 0.2989 * ref[..., 0] + 0.5870 * ref[..., 1] + 0.1140 * ref[..., 2]

        return mix, ref

    def __call__(self, mixture, ref):
        if isinstance(mixture, torch.Tensor):
            mixture = mixture.numpy()
        if isinstance(ref, torch.Tensor):
            ref = ref.numpy()
        mix, ref_np = self._prepare_inputs(mixture, ref)
        out = self.sess.run(None, {"mixture": mix, "ref": ref_np})
        return torch.from_numpy(out[0])

    def call_numpy(self, mixture_np, ref_np):
        """纯 numpy 输入/输出，避免 torch tensor 中转。"""
        mix, ref_np = self._prepare_inputs(mixture_np, ref_np)
        out = self.sess.run(None, {"mixture": mix, "ref": ref_np})
        return out[0]

    def clear_stream_cache(self):
        pass

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


class _RKNNORTSplitModelWrapper:
    """Split inference: RKNN runs separator (mixture+ref→sep_pack), ORT runs decoder.

    RGB→grayscale conversion is done in CPU as preprocessing before feeding to RKNN.
    """

    def __init__(self, rknn_sep_path: str, decoder_onnx_path: str,
                 audio_len: int = 9600, ref_frames: int = 18,
                 image_size: int = 96, decoder_num_threads: int = 4):
        from rknn.api import RKNN
        import onnxruntime as ort

        # Load RKNN separator model
        self.rknn = RKNN(verbose=False)
        ret = self.rknn.load_rknn(rknn_sep_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        self.audio_len = int(audio_len)
        self.ref_frames = int(ref_frames)
        self.image_size = int(image_size)

        # Load ORT decoder
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if decoder_num_threads > 0:
            opts.intra_op_num_threads = decoder_num_threads
            opts.inter_op_num_threads = 1
        self.decoder_sess = ort.InferenceSession(
            decoder_onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # Read decoder input channel count from ONNX
        self.n_ch = 512
        for inp in self.decoder_sess.get_inputs():
            if inp.name == "mixture_w" and len(inp.shape) >= 2:
                ch = _onnx_dim_to_int(inp.shape[1])
                if ch is not None and ch > 0:
                    self.n_ch = int(ch)
                    break
        self.normalize_mode = "mossformer"

    @staticmethod
    def _fit_time_axis(arr: np.ndarray, axis: int, target_len: int) -> np.ndarray:
        cur = int(arr.shape[axis])
        tgt = int(target_len)
        if cur == tgt:
            return arr.astype(np.float32, copy=False)
        if cur > tgt:
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(0, tgt)
            return arr[tuple(sl)].astype(np.float32, copy=False)
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, tgt - cur)
        return np.pad(arr.astype(np.float32, copy=False), pad_width, mode="constant")

    def _prepare_inputs(self, mixture_np: np.ndarray, ref_np: np.ndarray):
        mix = np.asarray(mixture_np, dtype=np.float32)
        ref = np.asarray(ref_np, dtype=np.float32)
        # Fit to fixed lengths
        if self.audio_len > 0:
            mix = self._fit_time_axis(mix, 1, self.audio_len)
        if self.ref_frames > 0:
            ref = self._fit_time_axis(ref, 1, self.ref_frames)
        # RGB→grayscale if 5D input
        if self.normalize_mode == "mossformer" and ref.ndim == 5 and ref.shape[-1] == 3:
            ref = 0.2989 * ref[..., 0] + 0.5870 * ref[..., 1] + 0.1140 * ref[..., 2]
        return mix, ref

    def __call__(self, mixture, ref):
        if isinstance(mixture, torch.Tensor):
            mixture = mixture.numpy()
        if isinstance(ref, torch.Tensor):
            ref = ref.numpy()
        return torch.from_numpy(self.call_numpy(mixture, ref))

    def call_numpy(self, mixture_np: np.ndarray, ref_np: np.ndarray) -> np.ndarray:
        mix, ref_gray = self._prepare_inputs(mixture_np, ref_np)
        # RKNN separator: mixture + ref_gray → sep_pack
        sep_pack = self.rknn.inference(inputs=[mix, ref_gray])[0]
        # Split sep_pack into mixture_w and est_mask along channel axis
        n_ch = self.n_ch
        mixture_w = sep_pack[:, :n_ch, :]
        est_mask = sep_pack[:, n_ch:, :]
        # ORT decoder
        t_audio = np.array([int(mix.shape[1])], dtype=np.int64)
        out = self.decoder_sess.run(
            None,
            {"mixture_w": mixture_w.astype(np.float32),
             "est_mask": est_mask.astype(np.float32),
             "target_audio_len": t_audio}
        )
        return out[0].squeeze(0)

    def clear_stream_cache(self):
        pass

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


class _TorchScriptModelWrapper:
    """TorchScript (LibTorch C++ 部署) 推理包装，接口与 PyTorch model 一致。"""

    def __init__(self, ts_path: str, device: torch.device):
        self.model = torch.jit.load(ts_path, map_location=device)
        self.model.eval()
        self.model = torch.jit.optimize_for_inference(self.model)
        if int(os.environ.get("JIT_OPTIM_AT_LOAD", "1") or 0):
            try:
                self.model = torch.jit.freeze(self.model)
            except Exception:
                pass
        self.device = device


    def __call__(self, mixture, ref):
        with torch.no_grad():
            return self.model(mixture, ref)

    def clear_stream_cache(self):
        fn = getattr(self.model, "clear_stream_cache", None)
        if callable(fn):
            fn()

    def trim_stream_cache(self, n_audio_drop: int, n_ref_drop: int):
        fn = getattr(self.model, "trim_stream_cache", None)
        if callable(fn):
            fn(n_audio_drop=n_audio_drop, n_ref_drop=n_ref_drop)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


def _apply_cpu_thread_limits(n: int) -> None:
    """限制 CPU 并行度（torch/cv2）并启用推理友好后端开关。"""
    n = int(n)
    if n <= 0:
        return
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass
    if int(os.environ.get("INFER_TUNE_CPU", "1") or 0):
        try:
            torch.backends.mkldnn.enabled = True
        except Exception:
            pass
        try:
            torch.backends.mkldnn.deterministic = False
        except Exception:
            pass
        try:
            torch._C._jit_set_profiling_executor(False)
        except Exception:
            pass
        try:
            torch._C._jit_set_profiling_mode(False)
        except Exception:
            pass
        try:
            torch.jit.set_fusion_strategy([("STATIC", 1)])
        except Exception:
            pass
        try:
            torch._C._jit_override_can_fuse_on_cpu(True)
        except Exception:
            pass
        try:
            torch.jit.enable_onednn_fusion(True)
        except Exception:
            pass


def _dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(v) for v in d]
    return d


def _strip_leading_module(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def _load_model_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(pretrained_model, dict) and pretrained_model:
        pretrained_model = {_strip_leading_module(k): v for k, v in pretrained_model.items()}
        ks = list(pretrained_model.keys())
        if ks and not any(k.startswith("av_skim.") or k.startswith("model.") for k in ks):
            if any(k.startswith("sep_network.") or k.startswith("ref_encoder.") for k in ks):
                pretrained_model = {f"model.{k}": v for k, v in pretrained_model.items()}

    state = model.state_dict()
    for key in list(state.keys()):
        bare = _strip_leading_module(key)
        picked = None
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            picked = pretrained_model[key]
        elif bare in pretrained_model and state[key].shape == pretrained_model[bare].shape:
            picked = pretrained_model[bare]
        elif f"module.{key}" in pretrained_model and state[key].shape == pretrained_model[f"module.{key}"].shape:
            picked = pretrained_model[f"module.{key}"]
        elif f"module.{bare}" in pretrained_model and state[key].shape == pretrained_model[f"module.{bare}"].shape:
            picked = pretrained_model[f"module.{bare}"]
        if picked is not None:
            state[key] = picked
    model.load_state_dict(state)


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


def _normalize_face_target_policy(policy: str) -> str:
    pol = str(policy).lower().strip()
    if pol.endswith("_lock"):
        return pol[: -len("_lock")]
    return pol


def pick_target_detection(
    boxes: Sequence[Tuple[float, np.ndarray, Optional[float], Optional[np.ndarray]]],
    frame_w: int,
    frame_h: int,
    policy: str = "largest",
    area_ratio_thr: float = 0.85,
    locked_box: Optional[np.ndarray] = None,
    lock_min_iou: float = 0.15,
) -> int:
    """从多人脸检测候选中选目标索引。元素为 (area, box_xyxy, score|None, lip_xy|None)。

    若提供 locked_box，则在候选中选与锁定框 IoU 最大者；若最大 IoU < lock_min_iou 则返回 -1（保持上一帧目标）。
    """
    if not boxes:
        return 0
    pol = _normalize_face_target_policy(policy)

    if locked_box is not None:
        locked = np.asarray(locked_box, dtype=np.float32).reshape(-1)
        best_i = 0
        best_iou = -1.0
        for i, b in enumerate(boxes):
            iou = _box_iou_xyxy(locked, b[1])
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if float(best_iou) >= float(lock_min_iou):
            return int(best_i)
        return -1

    if pol == "largest":
        return int(max(range(len(boxes)), key=lambda i: float(boxes[i][0])))

    cx_img = float(frame_w) * 0.5
    cy_img = float(frame_h) * 0.5

    def _center_dist_sq(i: int) -> float:
        box = boxes[i][1]
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        return (cx - cx_img) ** 2 + (cy - cy_img) ** 2

    if pol == "center":
        return int(min(range(len(boxes)), key=_center_dist_sq))

    if pol == "center_largest":
        max_area = max(float(b[0]) for b in boxes)
        thr = float(area_ratio_thr) * max_area
        candidates = [i for i, b in enumerate(boxes) if float(b[0]) >= thr - 1e-6]
        if not candidates:
            candidates = list(range(len(boxes)))
        return int(min(candidates, key=_center_dist_sq))

    raise ValueError(f"unsupported face target policy: {policy!r}")


class FaceHaarStreamTracker:
    """流式人脸框跟踪：OpenCV Haar + 平滑。"""

    def __init__(
        self,
        crop_size=128,
        face_scale=1.25,
        detect_every_n=5,
        detect_max_side=320,
        haar_scale_factor=1.15,
        haar_min_neighbors=4,
        box_smooth_alpha=0.85,
        target_policy: str = "center_largest",
        target_lock: bool = True,
        target_lock_min_iou: float = 0.15,
    ):
        self.target_policy = str(target_policy).lower().strip()
        self.target_lock = bool(target_lock) or self.target_policy.endswith("_lock")
        self.target_lock_min_iou = float(target_lock_min_iou)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.detector.empty():
            raise RuntimeError("加载 OpenCV haarcascade 失败")
        self.crop_size = int(crop_size)
        self.face_scale = float(face_scale)
        self.detect_every_n = max(1, int(detect_every_n))
        self.detect_max_side = int(detect_max_side)
        self.haar_scale_factor = float(haar_scale_factor)
        self.haar_min_neighbors = int(haar_min_neighbors)
        self.box_smooth_alpha = float(np.clip(box_smooth_alpha, 0.0, 1.0))
        self.last_box = None
        self.frame_idx = 0
        self.last_detected_box = None
        self.interferer_boxes: List = []
        self.interferer_box_scores: List[Optional[float]] = []
        self.target_score: Optional[float] = None

    _box_iou_xyxy = staticmethod(_box_iou_xyxy)

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
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            det_gray = gray
            sx = sy = 1.0
            if self.detect_max_side > 0:
                ms = max(h, w)
                if ms > self.detect_max_side:
                    scale = float(self.detect_max_side) / float(ms)
                    sw = max(1, int(round(w * scale)))
                    sh = max(1, int(round(h * scale)))
                    det_gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
                    sx = w / float(sw)
                    sy = h / float(sh)
            min_sz = max(20, int(30 * min(sx, sy)))
            faces = self.detector.detectMultiScale(
                det_gray,
                scaleFactor=self.haar_scale_factor,
                minNeighbors=self.haar_min_neighbors,
                minSize=(min_sz, min_sz),
            )
            if len(faces) > 0:
                scaled_boxes: List[Tuple[float, np.ndarray, Optional[float], Optional[np.ndarray]]] = []
                for fx, fy, fw, fh in faces:
                    x = int(round(fx * sx))
                    y = int(round(fy * sy))
                    fw = int(round(fw * sx))
                    fh = int(round(fh * sy))
                    cx = x + fw / 2.0
                    cy = y + fh / 2.0
                    s = max(fw, fh) * self.face_scale
                    x1 = int(round(cx - s / 2.0))
                    y1 = int(round(cy - s / 2.0))
                    x2 = int(round(cx + s / 2.0))
                    y2 = int(round(cy + s / 2.0))
                    box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                    scaled_boxes.append((area, box, None, None))
                ti = pick_target_detection(
                    scaled_boxes,
                    w,
                    h,
                    policy=self.target_policy,
                    locked_box=self._locked_box_for_pick(),
                    lock_min_iou=self.target_lock_min_iou,
                )
                if ti >= 0:
                    new_box = scaled_boxes[ti][1]
                    self.interferer_boxes = [
                        scaled_boxes[i][1].tolist() for i in range(len(scaled_boxes)) if i != ti
                    ]
                    self.interferer_box_scores = [None] * len(self.interferer_boxes)
                    if self.last_detected_box is not None and not self.target_lock:
                        iou = self._box_iou_xyxy(self.last_detected_box, new_box)
                        if float(iou) < float(scene_switch_iou_thr):
                            scene_switched = True
                    self.last_detected_box = new_box.copy()
                    self.target_score = None
                    if self.last_box is None or self.box_smooth_alpha >= 0.999:
                        self.last_box = new_box.tolist()
                    else:
                        prev = np.array(self.last_box, dtype=np.float32)
                        smoothed = self.box_smooth_alpha * prev + (1.0 - self.box_smooth_alpha) * new_box
                        self.last_box = smoothed.tolist()
                else:
                    self.interferer_boxes = [scaled_boxes[i][1].tolist() for i in range(len(scaled_boxes))]
                    self.interferer_box_scores = [None] * len(self.interferer_boxes)
            else:
                self.last_box = None
                self.last_detected_box = None
                self.target_score = None

        if self.last_box is None:
            self.frame_idx += 1
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32), bool(scene_switched)

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


def _draw_overlay_box_with_label(
    vis: np.ndarray,
    box: List[float],
    color_bgr: Tuple[int, int, int],
    score: Optional[float],
    font_scale: float = 1.6,
    box_thickness: int = 2,
    text_thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, box_thickness, lineType=cv2.LINE_AA)
    label = f"{score:.2f}" if score is not None else "--"
    ty = max(int(18 * font_scale), y1 - 4)
    cv2.putText(
        vis,
        label,
        (x1, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_bgr,
        text_thickness,
        lineType=cv2.LINE_AA,
    )


def _overlay_font_params(out_w: int, out_h: int) -> Tuple[float, int, int]:
    """Larger text on bigger outputs; tuned for readability after downscale."""
    short = max(1, min(int(out_w), int(out_h)))
    font_scale = max(1.25, min(3.0, short / 220.0))
    box_th = max(2, int(round(font_scale)))
    txt_th = max(2, int(round(font_scale * 0.85)))
    return float(font_scale), box_th, txt_th


def _scale_overlay_box(box: List[float], sx: float, sy: float) -> List[float]:
    return [
        float(box[0]) * sx,
        float(box[1]) * sy,
        float(box[2]) * sx,
        float(box[3]) * sy,
    ]


def _to_video_index(sample_idx: int, audio_sr: int, ref_sr: float) -> int:
    return int(np.floor(float(sample_idx) / float(audio_sr) * float(ref_sr)))


def _hop_should_mute_separated_audio(
    cur_start: int,
    cur_end: int,
    audio_sr: int,
    ref_sr: float,
    vid_face_valid: Sequence[bool],
) -> bool:
    """True if any reference frame mapped to [cur_start, cur_end) is not face-valid."""
    if cur_end <= cur_start or len(vid_face_valid) == 0:
        return False
    n = len(vid_face_valid)
    vi0 = _to_video_index(cur_start, audio_sr, ref_sr)
    vi1 = _to_video_index(cur_end - 1, audio_sr, ref_sr)
    vi0 = max(0, min(vi0, n - 1))
    vi1 = max(vi0, min(vi1, n - 1))
    for vi in range(vi0, vi1 + 1):
        if not bool(vid_face_valid[vi]):
            return True
    return False


def _align_audio_video_list_with_valid(
    audio: np.ndarray,
    video_list: List[np.ndarray],
    face_valid: Optional[List[bool]],
    audio_sr: int,
    video_fps: float,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[List[bool]]]:
    if face_valid is not None and len(face_valid) != len(video_list):
        raise ValueError(
            f"face_valid length ({len(face_valid)}) must match video_list ({len(video_list)})"
        )
    max_audio = int(np.floor(len(video_list) / float(video_fps) * float(audio_sr)))
    if max_audio <= 0:
        raise RuntimeError("视频太短，无法对齐")
    min_audio = min(int(audio.shape[0]), max_audio)
    audio = audio[:min_audio]

    keep_video = int(np.floor(float(audio.shape[0]) / float(audio_sr) * float(video_fps)))
    keep_video = max(1, min(keep_video, len(video_list)))
    video_list = video_list[:keep_video]
    fv_out = None if face_valid is None else face_valid[:keep_video]

    exact_audio = int(np.floor(float(keep_video) / float(video_fps) * float(audio_sr)))
    exact_audio = max(1, min(exact_audio, int(audio.shape[0])))
    audio = audio[:exact_audio]
    return audio, video_list, fv_out


def _apply_av_offset_with_valid(
    audio: np.ndarray,
    video_list: List[np.ndarray],
    face_valid: Optional[List[bool]],
    audio_sr: int,
    video_fps: float,
    av_offset_ms: float,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[List[bool]]]:
    if face_valid is not None and len(face_valid) != len(video_list):
        raise ValueError(
            f"face_valid length ({len(face_valid)}) must match video_list ({len(video_list)})"
        )
    if abs(float(av_offset_ms)) < 1e-9:
        return audio, video_list, face_valid
    if audio.size == 0 or len(video_list) == 0:
        return audio, video_list, face_valid
    if av_offset_ms > 0:
        off_samples = int(round(float(av_offset_ms) * float(audio_sr) / 1000.0))
        off_samples = max(0, min(off_samples, int(audio.shape[0])))
        audio = audio[off_samples:]
        return audio, video_list, face_valid
    off_frames = int(round(abs(float(av_offset_ms)) * float(video_fps) / 1000.0))
    off_frames = max(0, min(off_frames, len(video_list)))
    video_list = video_list[off_frames:]
    fv2 = None if face_valid is None else face_valid[off_frames:]
    return audio, video_list, fv2


def _align_audio_video_list(audio: np.ndarray, video_list: List[np.ndarray], audio_sr: int, video_fps: float):
    max_audio = int(np.floor(len(video_list) / float(video_fps) * float(audio_sr)))
    if max_audio <= 0:
        raise RuntimeError("视频太短，无法对齐")
    min_audio = min(int(audio.shape[0]), max_audio)
    audio = audio[:min_audio]

    keep_video = int(np.floor(float(audio.shape[0]) / float(audio_sr) * float(video_fps)))
    keep_video = max(1, min(keep_video, len(video_list)))
    video_list = video_list[:keep_video]

    exact_audio = int(np.floor(float(keep_video) / float(video_fps) * float(audio_sr)))
    exact_audio = max(1, min(exact_audio, int(audio.shape[0])))
    audio = audio[:exact_audio]
    return audio, video_list


def _apply_av_offset(
    audio: np.ndarray,
    video_list: List[np.ndarray],
    audio_sr: int,
    video_fps: float,
    av_offset_ms: float,
):
    if abs(float(av_offset_ms)) < 1e-9:
        return audio, video_list
    if audio.size == 0 or len(video_list) == 0:
        return audio, video_list
    if av_offset_ms > 0:
        off_samples = int(round(float(av_offset_ms) * float(audio_sr) / 1000.0))
        off_samples = max(0, min(off_samples, int(audio.shape[0])))
        audio = audio[off_samples:]
    else:
        off_frames = int(round(abs(float(av_offset_ms)) * float(video_fps) / 1000.0))
        off_frames = max(0, min(off_frames, len(video_list)))
        video_list = video_list[off_frames:]
    return audio, video_list


class IncrementalVideoResampler:
    """把人脸 crop 流式累积并映射到 ref_sr 的时间轴。"""

    def __init__(
        self,
        src_fps: float,
        target_fps: float,
        image_size: int,
        mean: float,
        std: float,
        normalize_mode: str = "mossformer",
    ):
        self.src_fps = float(src_fps)
        self.target_fps = float(target_fps)
        self.image_size = int(image_size)
        self.mean = float(mean)
        self.std = float(std)
        self.normalize_mode = str(normalize_mode)
        self._src_frames: List[np.ndarray] = []
        self._src_face_valid: List[bool] = []
        self._tgt_frames: List[np.ndarray] = []
        self._tgt_face_valid: List[bool] = []
        self._src_dropped: int = 0
        self._tgt_dropped: int = 0
        self._tgt_stack_cache: Optional[np.ndarray] = None

    @property
    def tgt_frames(self) -> List[np.ndarray]:
        return self._tgt_frames

    @property
    def tgt_face_valid(self) -> List[bool]:
        return self._tgt_face_valid

    @property
    def tgt_dropped(self) -> int:
        return int(self._tgt_dropped)

    def _invalidate_tgt_stack_cache(self) -> None:
        self._tgt_stack_cache = None

    def tgt_video_np(self) -> np.ndarray:
        """将 tgt_frames 堆成 [T,H,W,C] float32，增量缓存避免每 hop 全量 np.stack。"""
        n = len(self._tgt_frames)
        if n <= 0:
            self._tgt_stack_cache = None
            return np.zeros((0, self.image_size, self.image_size, 3), dtype=np.float32)
        cache = self._tgt_stack_cache
        if cache is not None and int(cache.shape[0]) == n:
            return cache
        if cache is not None and int(cache.shape[0]) < n:
            tail = np.stack(self._tgt_frames[int(cache.shape[0]) :], axis=0).astype(np.float32, copy=False)
            if tail.size > 0:
                self._tgt_stack_cache = np.concatenate([cache, tail], axis=0)
                return self._tgt_stack_cache
        self._tgt_stack_cache = np.stack(self._tgt_frames, axis=0).astype(np.float32, copy=False)
        return self._tgt_stack_cache

    def append_src_faces_rgb255(self, new_src_frames: List[np.ndarray], src_face_valid: Optional[List[bool]] = None):
        if not new_src_frames:
            return
        if new_src_frames:
            self._invalidate_tgt_stack_cache()
        if src_face_valid is None:
            src_face_valid = [True] * len(new_src_frames)
        elif len(src_face_valid) != len(new_src_frames):
            raise ValueError("src_face_valid length must match new_src_frames")
        for frm, ok in zip(new_src_frames, src_face_valid):
            self._src_frames.append(frm)
            self._src_face_valid.append(bool(ok))

        src_len_abs = self._src_dropped + len(self._src_frames)
        duration_sec = max(src_len_abs / self.src_fps, 1.0 / max(self.src_fps, 1.0))
        tgt_len_total_abs = max(1, int(np.round(duration_sec * self.target_fps)))
        tgt_len_prev_abs = self._tgt_dropped + len(self._tgt_frames)
        if tgt_len_total_abs <= tgt_len_prev_abs:
            return

        for ti_abs in range(tgt_len_prev_abs, tgt_len_total_abs):
            si_abs = int(
                np.clip(np.round((float(ti_abs) / self.target_fps) * self.src_fps), 0, src_len_abs - 1)
            )
            si_local = si_abs - self._src_dropped
            si_local = int(np.clip(si_local, 0, len(self._src_frames) - 1))
            frm = (self._src_frames[si_local].astype(np.float32, copy=False) / 255.0)
            v_ok = bool(self._src_face_valid[si_local])
            if frm.shape[0] != self.image_size or frm.shape[1] != self.image_size:
                frm = cv2.resize(frm, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            if self.normalize_mode == "mossformer":
                frm = (frm - self.mean) / self.std
            self._tgt_frames.append(frm.astype(np.float32, copy=False))
            self._tgt_face_valid.append(v_ok)

    def trim_head(self, n_target_frames: int) -> int:
        """Drop n target frames from ring buffer head, and drop matched source frames."""
        n_t = int(n_target_frames)
        if n_t <= 0:
            return 0
        n_t = min(n_t, len(self._tgt_frames))
        if n_t <= 0:
            return 0
        new_tgt_dropped_abs = self._tgt_dropped + n_t
        target_min_si_abs = int(
            math.floor(new_tgt_dropped_abs / max(self.target_fps, 1e-9) * self.src_fps)
        ) - 2
        target_min_si_abs = max(0, target_min_si_abs)
        n_s = max(0, target_min_si_abs - self._src_dropped)
        n_s = min(n_s, len(self._src_frames))

        del self._tgt_frames[:n_t]
        del self._tgt_face_valid[:n_t]
        self._tgt_dropped += n_t
        if self._tgt_stack_cache is not None and n_t > 0:
            self._tgt_stack_cache = self._tgt_stack_cache[n_t:].copy()
        if n_s > 0:
            del self._src_frames[:n_s]
            del self._src_face_valid[:n_s]
            self._src_dropped += n_s
        return n_t


def _video_buffer_len(vid_norm_list) -> int:
    if isinstance(vid_norm_list, np.ndarray):
        return int(vid_norm_list.shape[0])
    return len(vid_norm_list)


def _stack_video_for_model(vid_norm_list, use_numpy_path: bool, device) -> Any:
    if isinstance(vid_norm_list, np.ndarray):
        vid_np = vid_norm_list.astype(np.float32, copy=False)
        if use_numpy_path:
            return vid_np[np.newaxis, :]
        return torch.from_numpy(vid_np).unsqueeze(0).to(device, non_blocking=True)
    if use_numpy_path:
        return np.stack(vid_norm_list, axis=0).astype(np.float32, copy=False)[np.newaxis, :]
    vid_full_np = np.stack(vid_norm_list, axis=0)
    if vid_full_np.dtype != np.float32:
        vid_full_np = vid_full_np.astype(np.float32, copy=False)
    return torch.from_numpy(vid_full_np).unsqueeze(0).to(device, non_blocking=True)


def _run_new_hops_nonoverlap(
    wav_al: np.ndarray,
    vid_norm_list,
    model,
    device,
    hop_samples: int,
    context_samples: int,
    lookahead_samples: int,
    audio_sr: int,
    ref_sr: float,
    produced_samples: int,
    use_stream_cache: bool = False,
    vid_face_valid: Optional[Sequence[bool]] = None,
):
    new_segments = []
    total_samples = int(wav_al.shape[0])
    vid_len = _video_buffer_len(vid_norm_list)
    if total_samples <= 0 or vid_len == 0:
        return new_segments, int(produced_samples)
    if vid_face_valid is not None and len(vid_face_valid) != vid_len:
        raise ValueError(
            f"vid_face_valid length ({len(vid_face_valid)}) must match video buffer ({vid_len})"
        )

    # ONNX 后端使用纯 numpy 路径，避免 torch 转换开销
    use_numpy_path = hasattr(model, "call_numpy")

    if use_numpy_path:
        wav_full = wav_al.astype(np.float32, copy=False).reshape(1, -1)
        vid_full = _stack_video_for_model(vid_norm_list, True, device)
    else:
        wav_full_t = torch.from_numpy(wav_al)
        if wav_full_t.dtype != torch.float32:
            wav_full_t = wav_full_t.float()
        wav_full_t = wav_full_t.unsqueeze(0).to(device, non_blocking=True)
        vid_full_t = _stack_video_for_model(vid_norm_list, False, device)

    if use_stream_cache:
        end_emit = max(int(produced_samples), total_samples - max(0, int(lookahead_samples)))
        if end_emit <= int(produced_samples):
            return new_segments, int(produced_samples)
        if use_numpy_path:
            y_full = model.call_numpy(wav_full, vid_full).squeeze().astype(np.float32, copy=False)
        else:
            with torch.no_grad():
                y_full = model(wav_full_t, vid_full_t).squeeze().detach().cpu().numpy().astype(np.float32)
        p = int(produced_samples)
        while p < end_emit:
            cur_end = min(p + hop_samples, end_emit)
            target_len = cur_end - p
            seg = y_full[p:cur_end]
            if seg.shape[0] < target_len:
                seg = np.pad(seg, (0, target_len - seg.shape[0]), mode="constant")
            elif seg.shape[0] > target_len:
                seg = seg[:target_len]
            if vid_face_valid is not None and _hop_should_mute_separated_audio(
                p, cur_end, audio_sr, ref_sr, vid_face_valid
            ):
                seg = np.zeros(target_len, dtype=np.float32)
            new_segments.append(seg)
            p = cur_end
        return new_segments, p

    p = int(produced_samples)
    while p < total_samples:
        cur_start = p
        cur_end = min(cur_start + hop_samples, total_samples)
        win_start = max(0, cur_start - context_samples)
        win_end = min(total_samples, cur_end + lookahead_samples)

        v_start = _to_video_index(win_start, audio_sr, ref_sr)
        v_end = int(np.ceil(float(win_end) / float(audio_sr) * ref_sr))
        if use_numpy_path:
            v_start = max(0, min(v_start, vid_full.shape[1] - 1))
            v_end = max(v_start + 1, min(v_end, vid_full.shape[1]))
        else:
            v_start = max(0, min(v_start, int(vid_full_t.shape[1]) - 1))
            v_end = max(v_start + 1, min(v_end, int(vid_full_t.shape[1])))
        if win_end <= win_start or v_end <= v_start:
            break

        if use_numpy_path:
            a_in = wav_full[:, win_start:win_end]
            r_in = vid_full[:, v_start:v_end]
            y_win = model.call_numpy(a_in, r_in).squeeze().astype(np.float32, copy=False)
        else:
            a_in = wav_full_t[:, win_start:win_end]
            r_in = vid_full_t[:, v_start:v_end]
            with torch.no_grad():
                y_win = model(a_in, r_in).squeeze().detach().cpu().numpy().astype(np.float32)

        seg_local_start = max(0, cur_start - win_start)
        seg_local_end = max(seg_local_start, min(cur_end - win_start, y_win.shape[0]))
        seg = y_win[seg_local_start:seg_local_end]
        target_len = cur_end - cur_start
        if seg.shape[0] < target_len:
            seg = np.pad(seg, (0, target_len - seg.shape[0]), mode="constant")
        elif seg.shape[0] > target_len:
            seg = seg[:target_len]
        if vid_face_valid is not None and _hop_should_mute_separated_audio(
            cur_start, cur_end, audio_sr, ref_sr, vid_face_valid
        ):
            seg = np.zeros(target_len, dtype=np.float32)
        new_segments.append(seg)
        p = cur_end

    return new_segments, p


def _warmup_ingress_forward(
    model,
    device,
    image_size: int,
    audio_sr: int,
    ref_sr: float,
    hop_samples: int,
    context_samples: int,
    lookahead_samples: int,
) -> None:
    n_vid = max(128, int(np.ceil(6.0 * float(ref_sr))))
    audio_len = int(np.floor(float(n_vid) / float(ref_sr) * float(audio_sr))) + 4096
    audio = np.zeros(max(audio_len, context_samples + hop_samples + lookahead_samples + 1024), dtype=np.float32)
    video_list = [np.zeros((image_size, image_size, 3), dtype=np.float32) for _ in range(n_vid)]
    wav_al, vid_al = _align_audio_video_list(audio, video_list, audio_sr, ref_sr)
    if int(wav_al.shape[0]) < hop_samples or len(vid_al) < 2:
        return
    with torch.no_grad():
        _run_new_hops_nonoverlap(
            wav_al,
            vid_al,
            model,
            device,
            hop_samples,
            context_samples,
            lookahead_samples,
            audio_sr,
            ref_sr,
            0,
        )
    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize()


class AVStreamInference:
    """项目内部 AV 流式推理核心。"""

    def __init__(
        self,
        config: str = "./checkpoints/AV_Mossformer/config.yaml",
        checkpoint_dir: str = "./checkpoints/AV_Mossformer",
        use_cuda_override: int = 0,
        causal: Optional[int] = 1,
        context_ms: float = 100.0,
        infer_chunk_ms: float = 200.0,
        lookahead_frames: int = 0,
        cpu_threads: int = 8,
        onnx_path: Optional[str] = None,
        onnx_num_threads: int = 8,
        ts_path: Optional[str] = None,
        rknn_sep_path: Optional[str] = None,
        decoder_onnx_path: Optional[str] = None,
        rknn_audio_len: int = 9600,
        rknn_ref_frames: int = 18,
        decoder_num_threads: int = 4,
        face_crop_size: int = 96,
        face_scale: float = 0.8,
        face_detect_every_n: int = 5,
        face_detect_max_side: int = 320,
        haar_scale_factor: float = 1.15,
        haar_min_neighbors: int = 4,
        face_box_smooth_alpha: float = 0.85,
        clear_cache_on_scene_switch: int = 1,
        scene_switch_iou_thr: float = 0.15,
        av_offset_ms: float = 0.0,
        ingress_warmup: int = 1,
        use_stream_cache: int = 1,
        max_history_ms: float = 200.0,
        save_face_video_path: Optional[str] = None,
        save_face_overlay_video_path: Optional[str] = None,
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
        self.backend_kind = "torch"
        self.face_target_policy = str(face_target_policy).lower().strip()
        self.face_target_lock = bool(int(face_target_lock))
        self.face_target_lock_min_iou = float(face_target_lock_min_iou)
        self.face_detector = str(face_detector).lower()
        self.face_detector_model_path = str(face_detector_model_path)
        self.mediapipe_use_lip_center_crop = bool(int(mediapipe_use_lip_center_crop))
        self.mediapipe_lip_crop_scale = float(mediapipe_lip_crop_scale)
        self.mediapipe_lip_crop_min_px = int(mediapipe_lip_crop_min_px)
        self.mediapipe_lip_crop_max_px = int(mediapipe_lip_crop_max_px)
        self._warned_stream_cache_fallback = False
        self.save_face_video_path = str(save_face_video_path) if save_face_video_path else None
        self.save_face_overlay_video_path = (
            str(save_face_overlay_video_path) if save_face_overlay_video_path else None
        )
        self._face_video_writer = None
        self.overlay_write_stride = max(1, int(overlay_write_stride))
        self.overlay_scale = float(np.clip(float(overlay_scale), 0.1, 1.0))
        self._overlay_queue: Optional[queue.Queue] = None
        self._overlay_worker: Optional[threading.Thread] = None
        self._overlay_video_frame_idx = 0
        self._overlay_dropped_frames = 0
        self._face_video_fps: float = 30.0
        self._rtf_prof_face_s = 0.0
        self._rtf_prof_model_s = 0.0
        self._rtf_prof_align_s = 0.0
        if int(cpu_threads) > 0:
            _apply_cpu_thread_limits(int(cpu_threads))

        with open(str(config), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.ns = _dict_to_ns(cfg)
        if not hasattr(self.ns, "evaluate_only"):
            self.ns.evaluate_only = 0
        if not hasattr(self.ns, "network_audio"):
            raise ValueError("config 缺少 network_audio")
        if not hasattr(self.ns.network_audio, "tcn_attractor"):
            self.ns.network_audio.tcn_attractor = 0

        self.backbone = str(getattr(self.ns.network_audio, "backbone", ""))
        if self.backbone not in {"av_skim", "av_mossformer2_tse"}:
            raise ValueError(f"不支持 backbone={self.backbone}")

        self.ns.use_cuda = int(getattr(self.ns, "use_cuda", 0))
        if int(use_cuda_override) in (0, 1):
            self.ns.use_cuda = int(use_cuda_override)
        self.ns.device = torch.device("cuda" if (self.ns.use_cuda and torch.cuda.is_available()) else "cpu")
        if causal is not None:
            self.ns.causal = int(causal)

        self.audio_sr = int(getattr(self.ns, "audio_sr"))
        self.ref_sr = float(getattr(self.ns, "ref_sr"))
        self.image_size = int(getattr(self.ns.network_audio, "image_size", 112))
        self.mean = 0.506362
        self.std = 0.272877
        self.fixed_chunk_ms = float(infer_chunk_ms)
        self.hop_samples = max(1, int(round(self.audio_sr * (self.fixed_chunk_ms / 1000.0))))
        self.context_samples = max(0, int(round(self.audio_sr * (float(context_ms) / 1000.0))))
        self.lookahead_samples = max(0, int(round(self.audio_sr * (float(lookahead_frames) / self.ref_sr))))
        self.use_stream_cache = bool(int(use_stream_cache))
        self.max_history_samples = max(
            self.hop_samples + self.lookahead_samples + 1024,
            int(round(self.audio_sr * (float(max_history_ms) / 1000.0))),
        )
        drop_unit_ms = 100.0
        self.drop_unit_samples = max(1, int(round(self.audio_sr * (drop_unit_ms / 1000.0))))
        self.drop_unit_ref = max(1, int(round(self.ref_sr * (drop_unit_ms / 1000.0))))
        encoder_stride = int(getattr(self.ns.network_audio, "encoder_kernel_size", 16)) // 2
        encoder_stride = max(1, encoder_stride)
        self.drop_unit_enc = max(1, self.drop_unit_samples // encoder_stride)

        if onnx_path:
            self.backend_kind = "onnx"
            self.model = _ONNXModelWrapper(onnx_path, num_threads=int(onnx_num_threads))
            if int(ingress_warmup) != 0:
                _warmup_ingress_forward(
                    self.model,
                    self.ns.device,
                    self.image_size,
                    self.audio_sr,
                    self.ref_sr,
                    self.hop_samples,
                    self.context_samples,
                    self.lookahead_samples,
                )
        elif rknn_sep_path:
            self.backend_kind = "rknn_split"
            self.model = _RKNNORTSplitModelWrapper(
                rknn_sep_path=rknn_sep_path,
                decoder_onnx_path=decoder_onnx_path or os.path.join(
                    str(checkpoint_dir), "av_mossformer2_decoder_fixed.onnx"
                ),
                audio_len=int(rknn_audio_len),
                ref_frames=int(rknn_ref_frames),
                image_size=int(self.image_size),
                decoder_num_threads=int(decoder_num_threads),
            )
            if int(ingress_warmup) != 0:
                _warmup_ingress_forward(
                    self.model,
                    self.ns.device,
                    self.image_size,
                    self.audio_sr,
                    self.ref_sr,
                    self.hop_samples,
                    self.context_samples,
                    self.lookahead_samples,
                )
        elif ts_path:
            self.backend_kind = "torch_jit"
            self.model = _TorchScriptModelWrapper(ts_path, device=self.ns.device)
            if int(ingress_warmup) != 0:
                _warmup_ingress_forward(
                    self.model,
                    self.ns.device,
                    self.image_size,
                    self.audio_sr,
                    self.ref_sr,
                    self.hop_samples,
                    self.context_samples,
                    self.lookahead_samples,
                )
        else:
            self.backend_kind = "torch"
            self.model = network_wrapper(self.ns).to(self.ns.device)
            self.model.eval()
            ckpt_weights_only = os.path.join(str(checkpoint_dir), "last_best_weights_only.pt")
            ckpt_default = os.path.join(str(checkpoint_dir), "last_best_checkpoint.pt")
            ckpt = ckpt_weights_only if os.path.isfile(ckpt_weights_only) else ckpt_default
            _load_model_weights(self.model, ckpt)
            if int(ingress_warmup) != 0:
                _warmup_ingress_forward(
                    self.model,
                    self.ns.device,
                    self.image_size,
                    self.audio_sr,
                    self.ref_sr,
                    self.hop_samples,
                    self.context_samples,
                    self.lookahead_samples,
                )

        self._model_supports_stream_trim = callable(getattr(self.model, "trim_stream_cache", None))
        self._model_supports_stream_clear = callable(getattr(self.model, "clear_stream_cache", None))
        internal_stream_cache = bool(
            int(getattr(self.ns.network_audio, "stream_cache_enable", 0) or 0)
        )
        # use_stream_cache=1 且模型无 trim：仅当 yaml 开启*内部* stream cache 时才降级为滑窗。
        # ONNX / 无内部 cache 的 torch：与 torch 一致走 full-buffer，不依赖 trim_stream_cache。
        if (
            self.use_stream_cache
            and internal_stream_cache
            and (not self._model_supports_stream_trim)
        ):
            self.use_stream_cache = False
            if not self._warned_stream_cache_fallback:
                print(
                    "[WARN] stream_cache_enable=1 but backend lacks trim_stream_cache; "
                    "fallback to use_stream_cache=0 (per-hop sliding window)."
                )
                self._warned_stream_cache_fallback = True

        onnx_fixed = False
        if self.backend_kind == "onnx":
            onnx_fixed = bool(getattr(self.model, "has_fixed_input", False))
        print(
            f"[infer] backend={self.backend_kind} use_stream_cache={int(self.use_stream_cache)} "
            f"infer_chunk_ms={self.fixed_chunk_ms} context_samples={self.context_samples} "
            f"hop_samples={self.hop_samples}"
            + (f" onnx_fixed_input={int(onnx_fixed)}" if self.backend_kind == "onnx" else "")
        )

        self.av_offset_ms = float(av_offset_ms)
        self.clear_cache_on_scene_switch = int(clear_cache_on_scene_switch)
        self.scene_switch_iou_thr = float(scene_switch_iou_thr)
        self._tracker_args = dict(
            crop_size=int(face_crop_size),
            face_scale=float(face_scale),
            detect_every_n=int(face_detect_every_n),
            detect_max_side=int(face_detect_max_side),
            haar_scale_factor=float(haar_scale_factor),
            haar_min_neighbors=int(haar_min_neighbors),
            box_smooth_alpha=float(face_box_smooth_alpha),
            target_policy=self.face_target_policy,
            target_lock=self.face_target_lock,
            target_lock_min_iou=self.face_target_lock_min_iou,
        )
        self._started = False
        self._face_video_fps = float(self.ref_sr)
        self.reset(reopen_face_video=False)

    def _open_face_video_writer(self, path: str, fps: float) -> None:
        self._close_face_video_writer()
        out_path = str(path)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        size = int(self._tracker_args["crop_size"])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (size, size))
        if not writer.isOpened():
            print(f"[WARN] cannot open face video writer: {out_path}")
            return
        self._face_video_writer = writer
        print(f"[face video] {out_path} @ {float(fps):.2f} fps, {size}x{size}")

    def _write_face_frame(self, face_rgb: np.ndarray) -> None:
        if self._face_video_writer is None:
            return
        frm = np.clip(face_rgb, 0.0, 255.0).astype(np.uint8)
        if frm.ndim == 2:
            frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
        else:
            frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        self._face_video_writer.write(frm)

    def _close_face_video_writer(self) -> None:
        if self._face_video_writer is not None:
            self._face_video_writer.release()
            self._face_video_writer = None

    def _overlay_worker_loop(self) -> None:
        writer = None
        out_path = str(self.save_face_overlay_video_path or "")
        write_fps = float(self._face_video_fps) / float(self.overlay_write_stride)
        scale = float(self.overlay_scale)
        while True:
            item = self._overlay_queue.get()
            if item is None:
                break
            frame_bgr, interferer_boxes, interferer_scores, last_box, target_score = item
            h, w = frame_bgr.shape[:2]
            out_w, out_h = w, h
            sx = sy = 1.0
            if writer is None:
                parent = os.path.dirname(out_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, write_fps, (out_w, out_h))
                if not writer.isOpened():
                    print(f"[WARN] cannot open face overlay video writer: {out_path}")
                    writer = None
                    self._overlay_queue.task_done()
                    continue
                print(
                    f"[face overlay] {out_path} @ {write_fps:.2f} fps, {out_w}x{out_h} "
                    f"(stride={self.overlay_write_stride}, scale={scale:.2f})"
                )
            vis = frame_bgr
            fs, box_t, txt_t = _overlay_font_params(out_w, out_h)
            for box, sc in zip(interferer_boxes, interferer_scores):
                sb = _scale_overlay_box(list(box), sx, sy)
                _draw_overlay_box_with_label(
                    vis, sb, (0, 255, 255), sc, fs, box_t, txt_t
                )
            if last_box is not None:
                sb = _scale_overlay_box(list(last_box), sx, sy)
                _draw_overlay_box_with_label(
                    vis, sb, (0, 255, 0), target_score, fs, box_t, txt_t
                )
            if writer is not None:
                writer.write(vis)
            self._overlay_queue.task_done()
        if writer is not None:
            writer.release()

    def _ensure_overlay_worker(self) -> None:
        if not self.save_face_overlay_video_path:
            return
        if self._overlay_worker is not None and self._overlay_worker.is_alive():
            return
        self._overlay_queue = queue.Queue(maxsize=32)
        self._overlay_worker = threading.Thread(
            target=self._overlay_worker_loop, name="face_overlay_writer", daemon=True
        )
        self._overlay_worker.start()

    def _write_overlay_frame(self, frame_bgr: np.ndarray) -> None:
        if not self.save_face_overlay_video_path:
            return
        self._overlay_video_frame_idx += 1
        if (self._overlay_video_frame_idx - 1) % self.overlay_write_stride != 0:
            return
        self._ensure_overlay_worker()
        if self._overlay_queue is None:
            return
        interferer_boxes = list(getattr(self.tracker, "interferer_boxes", []) or [])
        isc = getattr(self.tracker, "interferer_box_scores", None)
        if isc is None:
            interferer_scores = [None] * len(interferer_boxes)
        else:
            interferer_scores = list(isc)
            while len(interferer_scores) < len(interferer_boxes):
                interferer_scores.append(None)
        last_box = (
            list(self.tracker.last_box) if self.tracker.last_box is not None else None
        )
        target_score = getattr(self.tracker, "target_score", None)
        scale = float(self.overlay_scale)
        frame_payload = frame_bgr
        if scale < 0.999:
            h, w = frame_bgr.shape[:2]
            out_w = max(1, int(round(w * scale)))
            out_h = max(1, int(round(h * scale)))
            sx = out_w / float(max(1, w))
            sy = out_h / float(max(1, h))
            frame_payload = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            interferer_boxes = [_scale_overlay_box(list(b), sx, sy) for b in interferer_boxes]
            if last_box is not None:
                last_box = _scale_overlay_box(list(last_box), sx, sy)
        else:
            frame_payload = frame_bgr.copy()
        try:
            self._overlay_queue.put_nowait(
                (
                    frame_payload,
                    interferer_boxes,
                    interferer_scores,
                    last_box,
                    target_score,
                )
            )
        except queue.Full:
            self._overlay_dropped_frames += 1

    def _close_overlay_video_writer(self) -> None:
        if self._overlay_queue is not None:
            try:
                self._overlay_queue.put_nowait(None)
            except queue.Full:
                self._overlay_queue.put(None)
            if self._overlay_worker is not None:
                self._overlay_worker.join(timeout=120.0)
        if self._overlay_dropped_frames > 0:
            print(f"[face overlay] dropped {self._overlay_dropped_frames} frames (queue full)")
        self._overlay_queue = None
        self._overlay_worker = None
        self._overlay_video_frame_idx = 0
        self._overlay_dropped_frames = 0
        self._rtf_prof_face_s = 0.0
        self._rtf_prof_model_s = 0.0
        self._rtf_prof_align_s = 0.0

    def print_rtf_profile(self) -> None:
        total = self._rtf_prof_face_s + self._rtf_prof_model_s + self._rtf_prof_align_s
        if total <= 1e-9:
            print("[rtf profile] no samples recorded")
            return
        print(
            f"[rtf profile] face={self._rtf_prof_face_s:.3f}s ({100*self._rtf_prof_face_s/total:.1f}%) "
            f"model={self._rtf_prof_model_s:.3f}s ({100*self._rtf_prof_model_s/total:.1f}%) "
            f"align={self._rtf_prof_align_s:.3f}s ({100*self._rtf_prof_align_s/total:.1f}%) "
            f"total={total:.3f}s"
        )

    def reset(self, reopen_face_video: bool = True):
        self._close_face_video_writer()
        self._close_overlay_video_writer()
        if self.face_detector == "mediapipe":
            from face_mediapipe_tracker import FaceMediaPipeStreamTracker

            ta = self._tracker_args
            self.tracker = FaceMediaPipeStreamTracker(
                crop_size=ta["crop_size"],
                face_scale=ta["face_scale"],
                detect_every_n=ta["detect_every_n"],
                box_smooth_alpha=ta["box_smooth_alpha"],
                model_path=self.face_detector_model_path,
                use_lip_center_crop=self.mediapipe_use_lip_center_crop,
                lip_crop_scale=self.mediapipe_lip_crop_scale,
                lip_crop_min_px=self.mediapipe_lip_crop_min_px,
                lip_crop_max_px=self.mediapipe_lip_crop_max_px,
                target_policy=ta.get("target_policy", "center_largest"),
                target_lock=ta.get("target_lock", True),
                target_lock_min_iou=ta.get("target_lock_min_iou", 0.15),
            )
        else:
            self.tracker = FaceHaarStreamTracker(**self._tracker_args)
        self.resampler = IncrementalVideoResampler(
            src_fps=self.ref_sr,
            target_fps=self.ref_sr,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
            normalize_mode=("av_skim" if self.backbone == "av_skim" else "mossformer"),
        )
        self.audio_buf = np.array([], dtype=np.float32)
        self.produced_samples = 0
        self.n_ring_trims = 0
        self._started = True
        if hasattr(self.model, "clear_stream_cache"):
            self.model.clear_stream_cache()
        if reopen_face_video and self.save_face_video_path:
            self._open_face_video_writer(self.save_face_video_path, self._face_video_fps)

    def close(self):
        self._close_face_video_writer()
        self._close_overlay_video_writer()
        self._started = False
        if hasattr(self.model, "clear_stream_cache"):
            self.model.clear_stream_cache()

    def _normalize_audio_chunk(self, audio_chunk: Optional[np.ndarray], sampling_rate: int) -> np.ndarray:
        if audio_chunk is None:
            return np.array([], dtype=np.float32)
        arr = np.asarray(audio_chunk)
        if arr.ndim == 2:
            arr = np.mean(arr.astype(np.float32, copy=False), axis=0, dtype=np.float32)
        elif arr.ndim != 1:
            raise ValueError(f"audio_chunk 必须是 (T,) 或 (C,T)，当前={arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        sr_in = int(sampling_rate)
        if sr_in != int(self.audio_sr) and arr.size > 0:
            try:
                import torchaudio
                arr = torchaudio.functional.resample(
                    torch.from_numpy(arr), sr_in, self.audio_sr
                ).numpy().astype(np.float32)
            except ImportError:
                arr = librosa.resample(arr, orig_sr=sr_in, target_sr=int(self.audio_sr)).astype(np.float32, copy=False)
        return arr

    def _normalize_video_chunk(self, video_chunk: Optional[Any]) -> List[np.ndarray]:
        if video_chunk is None:
            return []
        if isinstance(video_chunk, list):
            frames = video_chunk
        else:
            arr = np.asarray(video_chunk)
            if arr.ndim == 3:
                frames = [arr]
            elif arr.ndim == 4:
                frames = [arr[i] for i in range(arr.shape[0])]
            else:
                raise ValueError(f"video_chunk 必须是 list/3D/4D，当前={arr.shape}")
        out: List[np.ndarray] = []
        for frm in frames:
            f = np.asarray(frm)
            if f.ndim != 3 or int(f.shape[2]) != 3:
                raise ValueError(f"单帧必须是 (H,W,3)，当前={f.shape}")
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            out.append(f)
        return out

    def stream_inference(
        self,
        audio_chunk: Optional[np.ndarray] = None,
        video_chunk: Optional[Any] = None,
        is_start: bool = False,
        is_end: bool = False,
        flush_buffer: bool = False,
        sampling_rate: Optional[int] = None,
        video_fps: Optional[float] = None,
    ) -> List[np.ndarray]:
        if bool(is_start) or not self._started:
            if bool(is_start):
                if video_fps is not None and float(video_fps) > 1e-3:
                    self._face_video_fps = float(video_fps)
                else:
                    self._face_video_fps = float(self.ref_sr)
            self.reset(reopen_face_video=bool(is_start))

        sr_in = int(self.audio_sr if sampling_rate is None else sampling_rate)
        audio_in = self._normalize_audio_chunk(audio_chunk, sampling_rate=sr_in)
        video_in = self._normalize_video_chunk(video_chunk)

        new_faces: List[np.ndarray] = []
        src_face_valid: List[bool] = []
        scene_switch_hits = 0
        t_face0 = time.perf_counter()
        for fbgr in video_in:
            face_rgb, scene_switched = self.tracker.process_bgr(
                fbgr,
                scene_switch_iou_thr=float(self.scene_switch_iou_thr),
            )
            new_faces.append(face_rgb)
            src_face_valid.append(self.tracker.last_box is not None)
            self._write_face_frame(face_rgb)
            self._write_overlay_frame(fbgr)
            scene_switch_hits += int(scene_switched)
        self._rtf_prof_face_s += time.perf_counter() - t_face0
        if int(self.clear_cache_on_scene_switch) != 0 and scene_switch_hits > 0 and hasattr(self.model, "clear_stream_cache"):
            self.model.clear_stream_cache()

        if new_faces:
            self.resampler.append_src_faces_rgb255(new_faces, src_face_valid)
        if audio_in.size > 0:
            self.audio_buf = (
                np.concatenate([self.audio_buf, audio_in.astype(np.float32, copy=False)])
                if self.audio_buf.size
                else audio_in.astype(np.float32, copy=False)
            )

        # Keep ring buffers bounded to avoid unbounded growth on long streams.
        # - stream-cache mode: trim both external ring and model internal caches.
        # - non-stream-cache mode (e.g. most TorchScript exports): trim only external ring.
        if (not self.use_stream_cache) and self.audio_buf.size > self.max_history_samples:
            n_drop_audio = int(self.audio_buf.size) - int(self.max_history_samples)
            if n_drop_audio > 0:
                n_drop_audio = min(n_drop_audio, int(self.audio_buf.size))
                self.audio_buf = self.audio_buf[n_drop_audio:].copy()
                n_drop_ref = int(round(float(n_drop_audio) / float(max(1, int(self.audio_sr))) * float(self.ref_sr)))
                if n_drop_ref > 0:
                    self.resampler.trim_head(n_drop_ref)
                self.produced_samples = max(0, int(self.produced_samples) - n_drop_audio)

        # In stream-cache mode, keep ring buffer bounded and trim model caches in sync.
        if self.use_stream_cache and self.audio_buf.size > self.max_history_samples:
            excess = int(self.audio_buf.size) - int(self.max_history_samples)
            units = excess // self.drop_unit_samples
            if units > 0:
                n_drop_audio = int(units * self.drop_unit_samples)
                n_drop_ref = int(units * self.drop_unit_ref)
                n_drop_enc = int(units * self.drop_unit_enc)
                if n_drop_audio < int(self.audio_buf.size):
                    self.audio_buf = self.audio_buf[n_drop_audio:].copy()
                else:
                    self.audio_buf = np.array([], dtype=np.float32)
                actually_dropped_ref = int(self.resampler.trim_head(n_drop_ref))
                if hasattr(self.model, "trim_stream_cache"):
                    self.model.trim_stream_cache(
                        n_audio_drop=n_drop_enc,
                        n_ref_drop=actually_dropped_ref,
                    )
                self.produced_samples = max(0, int(self.produced_samples) - n_drop_audio)
                self.n_ring_trims += 1

        outputs: List[np.ndarray] = []
        do_flush = bool(flush_buffer) or bool(is_end)
        if self.resampler.tgt_frames and self.audio_buf.size > 0:
            tfv = self.resampler.tgt_face_valid
            if len(tfv) != len(self.resampler.tgt_frames):
                raise RuntimeError("internal: tgt_face_valid length mismatch vs tgt_frames")
            t_align0 = time.perf_counter()
            wav_off, vid_off, fv_off = _apply_av_offset_with_valid(
                self.audio_buf,
                self.resampler.tgt_frames,
                list(tfv),
                self.audio_sr,
                self.ref_sr,
                float(self.av_offset_ms),
            )
            wav_al, vid_list_al, fv_al = _align_audio_video_list_with_valid(
                wav_off, vid_off, fv_off, self.audio_sr, self.ref_sr
            )
            self._rtf_prof_align_s += time.perf_counter() - t_align0
            # stream-cache: 必须与 fv_al 同长的对齐后视频，不能用未裁剪的 tgt_video_np()
            if self.use_stream_cache:
                if len(vid_list_al) > 0:
                    vid_for_model = np.stack(vid_list_al, axis=0).astype(np.float32, copy=False)
                else:
                    vid_for_model = np.zeros(
                        (0, self.image_size, self.image_size, 3), dtype=np.float32
                    )
            else:
                vid_for_model = vid_list_al
            if fv_al is not None and _video_buffer_len(vid_for_model) != len(fv_al):
                raise RuntimeError(
                    f"internal: vid_for_model length ({_video_buffer_len(vid_for_model)}) "
                    f"must match fv_al ({len(fv_al)})"
                )
            available = int(wav_al.shape[0]) - int(self.produced_samples)
            if available >= int(self.hop_samples) or (do_flush and available > 0):
                t_model0 = time.perf_counter()
                new_segs, self.produced_samples = _run_new_hops_nonoverlap(
                    wav_al,
                    vid_for_model,
                    self.model,
                    self.ns.device,
                    self.hop_samples,
                    self.context_samples,
                    self.lookahead_samples,
                    self.audio_sr,
                    self.ref_sr,
                    self.produced_samples,
                    use_stream_cache=self.use_stream_cache,
                    vid_face_valid=fv_al,
                )
                self._rtf_prof_model_s += time.perf_counter() - t_model0
                if new_segs:
                    outputs.extend([seg.astype(np.float32, copy=False) for seg in new_segs])

        if bool(is_end):
            self._close_face_video_writer()
            self._close_overlay_video_writer()
            self.reset(reopen_face_video=False)
        return outputs

