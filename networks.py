import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

EPS = 1e-8


class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args

        if args.network_audio.backbone == "av_skim":
            from models.av_skim.av_skim import av_skim

            self.av_skim = av_skim(args)
            self._define_lip_ref_encoder()
            self.model = None
        elif args.network_audio.backbone == "av_mossformer2_tse":
            from models.av_mossformer2_tse.av_mossformer2 import av_mossformer2

            # causal 由 yaml / 命令行 --causal 决定（1=因果，0=非因果）；预训练权重多为非因果，改 1 后需自训或核对权重
            # 子模块名用 model，便于加载 ClearerVoice 公开权重（state_dict 为 model.sep_network.* / model.ref_encoder.*）
            self.model = av_mossformer2(args)
            self.av_skim = None
            self.network_v = None
        else:
            raise NameError("Wrong network selection: use av_skim or av_mossformer2_tse")

    def _define_lip_ref_encoder(self):
        assert self.args.network_reference.cue == "lip"

        if self.args.network_reference.backbone == "blazenet64":
            from models.visual_frontend.blazenet64 import visualNet as Visual_encoder
        else:
            raise NameError("Wrong reference network selection")
        self.network_v = Visual_encoder(self.args)

    @staticmethod
    def _video_rgb_to_gray(ref_bt_hwc):
        """ref: [B, T, H, W, 3] -> [B, T, H, W]，与 MossFormer Visual_encoder 输入一致。"""
        x = ref_bt_hwc
        if x.dim() != 5:
            raise ValueError(f"expected [B,T,H,W,3], got {tuple(x.shape)}")
        return (
            0.2989 * x[..., 0]
            + 0.5870 * x[..., 1]
            + 0.1140 * x[..., 2]
        )

    def forward(self, mixture, ref=None, reference=None):
        bb = self.args.network_audio.backbone

        if bb == "av_skim":
            visual = ref.to(self.args.device)
            visual = transforms.functional.rgb_to_grayscale(
                visual.permute((0, 1, 4, 2, 3))
            ).squeeze(2)
            ymin, ymax, xmin, xmax = 15, 91, 27, 103
            visual = visual[:, :, ymin:ymax, xmin:xmax]
            visual = visual.clone().detach()
            visual = self.network_v(visual)
            return self.av_skim(mixture, visual, reference)

        if bb == "av_mossformer2_tse":
            st = getattr(self.args, "infer_forward_timing", None)
            if st is not None:
                t_pre0 = time.perf_counter()
            ref = ref.to(self.args.device)
            ref = self._video_rgb_to_gray(ref)
            h = getattr(self.args.network_audio, "mossformer_face_size", None) or getattr(
                self.args.network_audio, "image_size", 112
            )
            w = h
            if ref.shape[2] != h or ref.shape[3] != w:
                b, t, _, _ = ref.shape
                ref = ref.reshape(b * t, 1, ref.shape[2], ref.shape[3])
                ref = F.interpolate(ref, size=(h, w), mode="bilinear", align_corners=False)
                ref = ref.reshape(b, t, h, w)
            if st is not None:
                st["nw_preprocess"] = float(st.get("nw_preprocess", 0.0)) + (
                    time.perf_counter() - t_pre0
                )
                t_in0 = time.perf_counter()
            out = self.model(mixture, ref)
            if st is not None:
                st["nw_inner_model"] = float(st.get("nw_inner_model", 0.0)) + (
                    time.perf_counter() - t_in0
                )
            return out

        raise NameError("Wrong network selection")

    def clear_stream_cache(self):
        if self.args.network_audio.backbone == "av_mossformer2_tse" and self.model is not None:
            if hasattr(self.model, "clear_stream_cache"):
                self.model.clear_stream_cache()

    def trim_stream_cache(self, n_audio_drop: int = 0, n_ref_drop: int = 0):
        if self.args.network_audio.backbone == "av_mossformer2_tse" and self.model is not None:
            if hasattr(self.model, "trim_stream_cache"):
                return self.model.trim_stream_cache(int(n_audio_drop), int(n_ref_drop))
        return {"audio_drop": 0, "ref_drop": 0}



