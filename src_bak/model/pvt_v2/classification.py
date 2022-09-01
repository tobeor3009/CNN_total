import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import Model
from tensorflow.keras import layers

from .layers import LayerArchive, EncodeBlock, OverlapPatchEmbed
from .layers import custom_init_layer_norm, custom_init_dense, identity_layer


def PVT_V2_Classification(input_shape,
                          num_classes=1000, activation=None,
                          patch_size=16, conv_stride=4,
                          embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                          mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=custom_init_layer_norm,
                          depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):

    # stochastic depth decay rule
    dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
    cur = 0

    layer_archive = LayerArchive()
    for i in range(num_stages):
        patch_size = 7 if i == 0 else 3
        stride = conv_stride * 2 if i == 0 else conv_stride
        patch_embed = OverlapPatchEmbed(
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dims[i]
        )
        block_list = [EncodeBlock(dim=embed_dims[i],
                                  num_heads=num_heads[i],
                                  mlp_ratio=mlp_ratios[i],
                                  qkv_bias=qkv_bias,
                                  qk_scale=qk_scale,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[cur + j],
                                  norm_layer=norm_layer,
                                  sr_ratio=sr_ratios[i], linear=linear) for j in range(depths[i])]

        norm = norm_layer()
        cur += depths[i]

        setattr(layer_archive, f"patch_embed{i + 1}", patch_embed)
        setattr(layer_archive, f"encode_block{i + 1}", block_list)
        setattr(layer_archive, f"encode_norm{i + 1}", norm)
        # classification head
        head = custom_init_dense(
            num_classes) if num_classes > 0 else identity_layer

    input_tensor = layers.Input(shape=input_shape)
    input_shape = np.array(input_shape)
    for i in range(num_stages):
        patch_embed = getattr(layer_archive, f"patch_embed{i + 1}")
        block = getattr(layer_archive, f"encode_block{i + 1}")
        norm = getattr(layer_archive, f"encode_norm{i + 1}")
        if i == 0:
            x = patch_embed(input_tensor)
        else:
            x = patch_embed(x)
        input_shape //= conv_stride * 2 if i == 0 else conv_stride
        H, W, _ = input_shape
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        if i != num_stages - 1:
            x = layers.Reshape((H, W, embed_dims[i]))(x)
    x = keras_backend.mean(x, axis=-1)
    # x = layers.Flatten()(x)
    x = head(x)
    x = activation(x)

    model = Model(input_tensor, x)
    return model
