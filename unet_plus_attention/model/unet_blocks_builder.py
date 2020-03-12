from torch import nn

from unet_plus_attention.model.attention_block import SelfAttentionBlock


def vanila_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        nn.Conv2d(in_, out, 3, padding=1),
        nn.BatchNorm2d(out),
        nn.ReLU(),
        nn.Conv2d(out, out, 3, padding=1),
        nn.BatchNorm2d(out),
        nn.ReLU()
    ])


def attention_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        nn.Conv2d(in_, in_ // 4, kernel_size=1),
        nn.BatchNorm2d(in_ // 4),
        nn.ReLU(),
        SelfAttentionBlock(in_ // 4, out // 4, kernel_size=5, stride=1, padding=2, groups=4),
        nn.BatchNorm2d(out // 4),
        nn.ReLU(),
        nn.Conv2d(out // 4, out, kernel_size=1),
        nn.BatchNorm2d(out),
        nn.ReLU()
    ])


class AttentionWithSkipConnection(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_, in_ // 4, kernel_size=1),
            nn.BatchNorm2d(in_ // 4),
            nn.ReLU(),
            SelfAttentionBlock(in_ // 4, out // 4, kernel_size=5, stride=1, padding=2, groups=4),
            nn.BatchNorm2d(out // 4),
            nn.ReLU(),
            nn.Conv2d(out // 4, out, kernel_size=1),
            nn.BatchNorm2d(out),
            nn.ReLU()
        )
        if in_ != out:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_, out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out)
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.seq(x)
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return out


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_, out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_, in_, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=in_
            ),
            nn.ReLU(),
            nn.BatchNorm2d(in_),
            nn.Conv2d(in_, out, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


def attention_with_skip_connection_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        AttentionWithSkipConnection(in_, out)
    ])


def depthwise_separable_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        DepthWiseSeparableConv(in_, out, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out),
        DepthWiseSeparableConv(out, out, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out),
    ])


def depthwise_separable_block_light_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        DepthWiseSeparableConv(in_, out, padding=1),
    ])
