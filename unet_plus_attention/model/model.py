from unet_plus_attention.model.unet import Unet
from unet_plus_attention.model.unet_blocks_builder import *


def get_vanila_unet_model(in_dim: int, out_dim: int) -> nn.Module:
    module_builders = {
        'enc1': vanila_block_build,
        'enc2': vanila_block_build,
        'enc3': vanila_block_build,
        'enc4': vanila_block_build,
        'enc5': vanila_block_build,
        'dec1': vanila_block_build,
        'dec2': vanila_block_build,
        'dec3': vanila_block_build,
        'dec4': vanila_block_build,
    }
    return Unet(module_builders, in_dim, out_dim)


def get_unet_attention_decoder(in_dim: int, out_dim: int) -> nn.Module:
    module_builders = {
        'enc1': vanila_block_build,
        'enc2': vanila_block_build,
        'enc3': vanila_block_build,
        'enc4': vanila_block_build,
        'enc5': vanila_block_build,
        'dec1': attention_block_build,
        'dec2': attention_block_build,
        'dec3': attention_block_build,
        'dec4': attention_block_build,
    }
    return Unet(module_builders, in_dim, out_dim)
