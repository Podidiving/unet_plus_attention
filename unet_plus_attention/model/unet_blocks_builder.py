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