from tensorflow.keras.models import Model
from tensorflow.keras import layers
from . import swin_layers
from . import transformer_layers
from . import utils


def swin_classification_2d_base(input_tensor, embed_dim, depth, patch_size, num_heads, window_size, num_mlp, act="gelu", shift_window=True):

    # Dropout parameters
    mlp_drop_rate = 0.01  # Droupout after each MLP layer
    attn_drop_rate = 0.01  # Dropout after Swin-Attention
    # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    proj_drop_rate = 0.01
    drop_path_rate = 0.01  # Drop-path within skip-connections

    qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

    input_shape = input_tensor.shape[1:]
    num_patch_x = input_shape[0] // patch_size[0]
    num_patch_y = input_shape[1] // patch_size[1]

    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    # Extract patches from the input tensor
    X = transformer_layers.patch_extract(patch_size)(input_tensor)

    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x * num_patch_y,
                                           embed_dim)(X)

    # -------------------- Swin transformers -------------------- #
    # Stage 1: window-attention + Swin-attention + patch-merging

    for idx in range(depth):

        if idx % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads[idx],
                                             window_size=window_size[idx], shift_size=shift_size_temp, num_mlp=num_mlp, act=act,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate,
                                             name='swin_block{}'.format(idx))(X)
    # Patch-merging
    #    Pooling patch sequences. Half the number of patches (skip every two patches) and double the embedded dimensions
    X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
                                         embed_dim=embed_dim, name='down{}'.format(idx))(X)
    return X


def get_swin_classification_2d(input_shape, last_channel_num, embed_dim, depth,
                               patch_size, num_heads, window_size, num_mlp,
                               act="gelu", shift_window=True):
    IN = layers.Input(input_shape)
    X = swin_classification_2d_base(IN, embed_dim, depth, patch_size,
                                    num_heads, window_size, num_mlp,
                                    act=act, shift_window=shift_window)
    X = layers.GlobalAveragePooling1D()(X)
    # The output section
    OUT = layers.Dense(last_channel_num, activation='softmax')(X)
    # Model configuration
    model = Model(inputs=[IN, ], outputs=[OUT, ])
    return model
