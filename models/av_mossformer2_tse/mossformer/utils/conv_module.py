import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    """Depthwise 1-D convolution (groups=in_channels)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """Conformer-style depthwise conv module (residual)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 17,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
        causal: bool = False,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd for SAME padding"
        assert expansion_factor == 2, "Only supports expansion_factor=2"

        self.causal = causal
        self.kernel_size = kernel_size
        # 非因果：对称 SAME；因果：卷积内 padding=0，在 forward 里左侧补零
        _pad = 0 if causal else (kernel_size - 1) // 2
        self.depthwise = DepthwiseConv1d(
            in_channels, in_channels, kernel_size, stride=1, padding=_pad
        )
        self.transpose = Transpose(shape=(1, 2))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.transpose(inputs)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise(x)
        out = x.transpose(1, 2)
        out = self.dropout(out)
        return inputs + out

