from torch import nn


class BlockBaseClass(nn.Module):
    """
    Class to inherit from, when creating new blocks
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
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._groups = groups
        self._bias = bias

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def groups(self):
        return self._groups

    def forward(self, x):
        raise NotImplementedError


def check_dims(
        out_channels: int,
        groups: int,
        in_channels: int
):
    """
    Check, that dimensions in attention block is Ok
    :param out_channels:
    :param groups:
    :param in_channels:
    :return:
    """
    assert not out_channels % 2, \
        f"number of out channels ({out_channels})" \
        " must be divisible by 2"

    assert not out_channels % groups, \
        f"number of out channels ({out_channels})" \
        f" must be divisible by number of groups ({groups})"

    assert not in_channels % groups, \
        f"number of in channels ({in_channels})" \
        f" must be divisible by number of groups ({groups})"

    # this assertion for embeddings
    assert not (out_channels // groups) % 2, \
        f"number of out channels ({out_channels})" \
        f" divided by number of groups ({groups})" \
        f" must be divisible by 2"
