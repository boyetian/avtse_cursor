# Copyright 2020 Smeet Shah
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Visual_encoder(nn.Module):
    def __init__(self, args):
        super(Visual_encoder, self).__init__()
        self.args = args
        v_dim = int(getattr(self.args.network_reference, "emb_size", 256))
        visual_hidden = int(getattr(self.args.network_reference, "visual_conv_hidden", 512))
        visual_pw_groups = int(getattr(self.args.network_reference, "visual_pw_groups", 1))

        # visual frontend
        self.v_frontend = VisualFrontend(args)
        resnet_out_channels = int(
            getattr(self.args.network_reference, "resnet_out_channels", self.v_frontend.resnet_out_channels)
        )
        self.v_ds = nn.Conv1d(resnet_out_channels, v_dim, 1, bias=False)

        # visual adaptor
        stacks = []
        for _ in range(5):
            stacks += [VisualConv1D(args, V=v_dim, H=visual_hidden, pw_groups=visual_pw_groups)]
        self.visual_conv = nn.Sequential(*stacks)

    def forward(self, visual):
        visual = self.v_frontend(visual.unsqueeze(1))
        visual = self.v_ds(visual)
        visual = self.visual_conv(visual)
        return visual


class ResNetLayer(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    def __init__(self, stage_channels, use_adaptive_pool=False):
        super(ResNet, self).__init__()
        if len(stage_channels) != 4:
            raise ValueError(f"stage_channels must have 4 entries, got {stage_channels}")
        c1, c2, c3, c4 = [int(v) for v in stage_channels]
        self.layer1 = ResNetLayer(c1, c1, stride=1)
        self.layer2 = ResNetLayer(c1, c2, stride=2)
        self.layer3 = ResNetLayer(c2, c3, stride=2)
        self.layer4 = ResNetLayer(c3, c4, stride=2)
        if bool(use_adaptive_pool):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            # Legacy path (matches old checkpoints trained with image_size=112).
            self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class VisualFrontend(nn.Module):
    def __init__(self, args):
        super(VisualFrontend, self).__init__()
        self.args = args
        nr = getattr(self.args, "network_reference", None)
        stage_channels = list(getattr(nr, "resnet_stage_channels", [64, 128, 256, 512]))
        if len(stage_channels) != 4:
            raise ValueError(f"network_reference.resnet_stage_channels must have 4 values, got {stage_channels}")
        conv3d_kernel = tuple(getattr(nr, "conv3d_kernel", [5, 7, 7]))
        if len(conv3d_kernel) != 3:
            raise ValueError(f"network_reference.conv3d_kernel must have 3 values, got {conv3d_kernel}")
        use_adaptive_pool = int(getattr(nr, "use_adaptive_pool", 0)) == 1
        kt, kh, kw = [int(v) for v in conv3d_kernel]
        c1 = int(stage_channels[0])
        self.resnet_out_channels = int(stage_channels[-1])
        if self.args.causal:
            padding = (kt - 1, kh // 2, kw // 2)
        else:
            padding = (kt // 2, kh // 2, kw // 2)
        self.temporal_trim = (kt - 1) if self.args.causal else 0

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=(kt, kh, kw), stride=(1, 2, 2), padding=padding, bias=False),
            nn.BatchNorm3d(c1, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet(stage_channels=stage_channels, use_adaptive_pool=use_adaptive_pool)

    def forward(self, batch):
        batchsize = batch.shape[0]
        batch = self.frontend3D[0](batch)
        if self.temporal_trim > 0:
            batch = batch[:, :, :-self.temporal_trim, :, :]
        batch = self.frontend3D[1](batch)
        batch = self.frontend3D[2](batch)
        batch = self.frontend3D[3](batch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, self.resnet_out_channels)
        outputBatch = outputBatch.transpose(1, 2)
        return outputBatch


class VisualConv1D(nn.Module):
    def __init__(self, args, V=256, H=512, kernel_size=3, dilation=1, pw_groups=1):
        super(VisualConv1D, self).__init__()
        self.args = args
        self.pw_groups = int(max(1, pw_groups))
        if (V % self.pw_groups) != 0 or (H % self.pw_groups) != 0:
            raise ValueError(
                f"visual pointwise groups={self.pw_groups} requires V and H divisible, got V={V}, H={H}"
            )

        self.relu_0 = nn.ReLU()
        self.norm_0 = nn.BatchNorm1d(V)
        self.conv1x1 = nn.Conv1d(V, H, 1, bias=False, groups=self.pw_groups)
        self.relu = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(H)
        self.dconv_pad = (dilation * (kernel_size - 1)) // 2 if not self.args.causal else (dilation * (kernel_size - 1))
        self.dsconv = nn.Conv1d(H, H, kernel_size, stride=1, padding=self.dconv_pad, dilation=1, groups=H)
        self.prelu = nn.PReLU()
        self.norm_2 = nn.BatchNorm1d(H)
        self.pw_conv = nn.Conv1d(H, V, 1, bias=False, groups=self.pw_groups)

    def forward(self, x):
        out = self.relu_0(x)
        out = self.norm_0(out)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = self.norm_1(out)
        out = self.dsconv(out)
        if self.args.causal:
            out = out[:, :, :-self.dconv_pad]
        out = self.prelu(out)
        out = self.norm_2(out)
        out = self.pw_conv(out)
        return out + x

