## Unet plus self attention

---

Implementation of vanilla [unet](https://arxiv.org/abs/1505.04597) & unet with [self-attention](https://arxiv.org/abs/1906.05909)

In self-attention implementation `conv3x3` blocks are replaced by `conv1x1+self-attention+conv1x1`. See `unet_plus_attention/model/model.py` & `unet_plus_attention/model/unet_blocks_builder.py` for more details
