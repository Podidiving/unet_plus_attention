import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


from unet_plus_attention.model.utils.helpers import check_dims
from unet_plus_attention.model.utils.helpers import BlockBaseClass


class SelfAttentionBlock(BlockBaseClass):
    """
    Main class, that implements logic of attention block of the paper
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 1,
            groups: int = 1,
            bias: bool = False
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias
        )

        check_dims(self._out_channels, self._groups, self._in_channels)

        self.__h_embedding: nn.Parameter = nn.Parameter(
            torch.randn(
                self._groups,
                (self._out_channels // self._groups) // 2,
                1, 1,
                self._kernel_size, 1
            ),
            requires_grad=True
        )
        self.__w_embedding: nn.Parameter = nn.Parameter(
            torch.randn(
                self._groups,
                (self._out_channels // self._groups) // 2,
                1, 1,
                1, self._kernel_size
            ),
            requires_grad=True
        )

        if self._bias:
            self.__bias = nn.Parameter(
                torch.randn(1, self._out_channels, 1, 1),
                requires_grad=True
            )

        self.__key:  nn.Conv2d = self.__create_conv()
        self.__query: nn.Conv2d = self.__create_conv()
        self.__value: nn.Conv2d = self.__create_conv()

        self.reset_params()

    def __create_conv(self):
        return nn.Conv2d(
            self._in_channels,
            self._out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            groups=self._groups
        )

    def reset_params(self):
        init.kaiming_normal_(self.__key.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.__query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.__value.weight, mode='fan_out', nonlinearity='relu')

        if self._bias:
            bound = 1 / torch.sqrt(self._out_channels).item()
            init.uniform_(self.__bias, -bound, bound)

        init.normal_(self.__w_embedding, 0, 1)
        init.normal_(self.__h_embedding, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        x_padded: torch.Tensor = F.pad(
            x, [
                self._padding,
                self._padding,
                self._padding,
                self._padding
            ]
        )

        # [B, C_in, H, W] -> [B, C_out, H, W]
        query_out: torch.Tensor = self.__query(x)
        # [B, C_in, H, W] -> [B, C_out, H + 2*padding, W + 2*padding]
        value_out: torch.Tensor = self.__value(x_padded)
        key_out: torch.Tensor = self.__key(x_padded)

        # (H + 2*padding - kernel_size) // stride + 1 == H
        # (W + 2*padding - kernel_size) // stride + 1 == W
        # [B, C_out, H + 2*padding, W + 2*padding] ->
        # [B, C_out, H, W, kernel_size, kernel_size]
        value_out = value_out.unfold(2, self._kernel_size, self._stride)
        value_out = value_out.unfold(3, self._kernel_size, self._stride)
        key_out = key_out.unfold(2, self._kernel_size, self._stride)
        key_out = key_out.unfold(3, self._kernel_size, self._stride)

        # [B, C_out, H, W, kernel_size, kernel_size] ->
        # -> [B, groups, C_out // groups, H, W, kernel_size**2]
        value_out = value_out.contiguous().view(
            batch,
            self._groups,
            self._out_channels // self._groups,
            height,
            width,
            self._kernel_size ** 2
        )
        key_out = key_out.contiguous().view(
            batch,
            self._groups,
            self._out_channels // self._groups,
            height,
            width,
            self._kernel_size ** 2
        )

        # [B, C_out, H, W] ->
        # [B, groups, C_out // groups, H, W, 1]
        query_out = query_out.contiguous().view(
            batch,
            self._groups,
            self._out_channels // self._groups,
            height,
            width,
            1
        )

        # [groups, C_out // 2 // groups, 1, 1, kernel_size, 1] +
        # [groups, C_out // 2 // groups, 1, 1, 1, kernel_size] ->
        # [groups, C_out // groups, 1, 1, kernel_size, kernel_size]
        hw_emb: torch.Tensor = torch.cat(
            [
                self.__h_embedding.repeat(1, 1, 1, 1, 1, self._kernel_size),
                self.__w_embedding.repeat(1, 1, 1, 1, self._kernel_size, 1)
            ],
            dim=1
        )
        # [groups, C_out // groups, 1, 1, kernel_size, kernel_size] ->
        # [groups, C_out // groups, 1, 1, kernel_size**2]
        hw_emb = hw_emb.contiguous().view(
            self._groups,
            self._out_channels // self._groups,
            1, 1,
            self._kernel_size**2
        )

        plus = key_out + hw_emb
        # -> [B, groups, C_out // groups, H, W, kernel_size**2]
        out: torch.Tensor = query_out * plus
        out = F.softmax(out, dim=-1)

        # -> [B, groups, C_out // groups, H, W]
        out = torch.einsum("bgchwk,bgchwk -> bgchw", [out, value_out])

        # -> [B, C_out, H, W]
        out = out.view(batch, self._out_channels, height, width)
        if self._bias:
            out += self.__bias

        return out
