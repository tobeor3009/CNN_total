from functools import partial
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.activations import tanh, gelu, softmax, sigmoid

USE_CONV_BIAS = True
USE_DENSE_BIAS = True
kaiming_initializer = tf.keras.initializers.HeNormal()


class GCBlock2D(layers.Layer):
    def __init__(self, in_channel, ratio=1, fusion_types=('channel_add',), **kwargs):
        super().__init__(**kwargs)
        assert in_channel is not None, 'GCBlock needs in_channel'
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channel = in_channel
        self.ratio = ratio
        self.middle_channel = int(in_channel * ratio)
        self.fusion_types = fusion_types
        self.key_mask = layers.Conv2D(filters=in_channel,
                                      kernel_size=1,
                                      kernel_initializer=kaiming_initializer,
                                      padding="same",
                                      use_bias=USE_CONV_BIAS)
        self.value_mask = layers.Conv2D(filters=1,
                                        kernel_size=1,
                                        kernel_initializer=kaiming_initializer,
                                        padding="same",
                                        use_bias=USE_CONV_BIAS)
        self.softmax = partial(softmax, axis=-2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = Sequential(
                [layers.Conv2D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv2D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Sequential(
                [layers.Conv2D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv2D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_mul_conv = None

    def build(self, input_shape):
        _, self.H, self.W, self.C = input_shape

    def call(self, x):

        # x.shape: [B, H, W, C]
        # key_mask.shape: [B, H, W, C]
        key_mask = self.key_mask(x)
        # key_mask.shape: [B, (H * W), C]
        key_mask = layers.Reshape((self.H * self.W, self.C))(key_mask)
        # key_mask.shape: [B, C, (H * W)]
        key_mask = layers.Permute((2, 1))(key_mask)

        # value_mask.shape: [B, H, W, 1]
        value_mask = self.value_mask(x)
        # value_mask.shape: [B, (H * W), 1]
        value_mask = layers.Reshape((self.H * self.W, 1))(value_mask)
        value_mask = self.softmax(value_mask)

        # [B, C, (H * W)] @ [B, (H * W), 1]
        # context_mask.shape: [B, C, 1]
        context_mask = tf.matmul(key_mask, value_mask)
        # context_mask.shape: [B, 1, 1, C]
        context_mask = layers.Reshape((1, 1, self.C))(context_mask)

        out = x
        if self.channel_mul_conv is not None:
            # [B, 1, 1, C]
            channel_mul_term = sigmoid(self.channel_mul_conv(context_mask))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [B, 1, 1, C]
            channel_add_term = self.channel_add_conv(context_mask)
            out = out + channel_add_term
        # out.shape: [B, H, W, C]
        return out


class GCBlock3D(layers.Layer):
    def __init__(self, in_channel, ratio=1, fusion_types=('channel_add',), **kwargs):
        super().__init__(**kwargs)
        assert in_channel is not None, 'GCBlock needs in_channel'
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channel = in_channel
        self.ratio = ratio
        self.middle_channel = int(in_channel * ratio)
        self.fusion_types = fusion_types

        self.key_mask = layers.Conv3D(filters=in_channel,
                                      kernel_size=1,
                                      kernel_initializer=kaiming_initializer,
                                      padding="same",
                                      use_bias=USE_CONV_BIAS)
        self.value_mask = layers.Conv3D(filters=1,
                                        kernel_size=1,
                                        kernel_initializer=kaiming_initializer,
                                        padding="same",
                                        use_bias=USE_CONV_BIAS)
        self.softmax = partial(softmax, axis=-2)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = Sequential(
                [layers.Conv3D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv3D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Sequential(
                [layers.Conv3D(self.middle_channel, kernel_size=1, use_bias=USE_CONV_BIAS),
                 layers.LayerNormalization(axis=-1),
                 layers.ReLU(max_value=6),  # yapf: disable
                 layers.Conv3D(self.in_channel, kernel_size=1, use_bias=USE_CONV_BIAS)])
        else:
            self.channel_mul_conv = None

    def build(self, input_shape):
        _, self.H, self.W, self.Z, self.C = input_shape

    def call(self, x):

        # x.shape: [B, H, W, Z, C]
        # key_mask.shape: [B, H, W, Z, C]
        key_mask = self.key_mask(x)
        # key_mask.shape: [B, (H * W * Z), C]
        key_mask = layers.Reshape((self.H * self.W * self.Z, self.C))(key_mask)
        # key_mask.shape: [B, C, (H * W)]
        key_mask = layers.Permute((2, 1))(key_mask)

        # value_mask.shape: [B, H, W, Z, 1]
        value_mask = self.value_mask(x)
        # value_mask.shape: [B, (H * W * Z), 1]
        value_mask = layers.Reshape((self.H * self.W * self.Z, 1))(value_mask)
        value_mask = self.softmax(value_mask)

        # [B, C, (H * W * Z)] @ [B, (H * W * Z), 1]
        # context_mask.shape: [B, C, 1]
        context_mask = tf.matmul(key_mask, value_mask)
        # context_mask.shape: [B, 1, 1, 1, C]
        context_mask = layers.Reshape((1, 1, 1, self.C))(context_mask)

        out = x
        if self.channel_mul_conv is not None:
            # [B, 1, 1, 1, C]
            channel_mul_term = sigmoid(self.channel_mul_conv(context_mask))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [B, 1, 1, 1, C]
            channel_add_term = self.channel_add_conv(context_mask)
            out = out + channel_add_term
        # out.shape: [B, H, W, Z, C]
        return out


def conv3d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              include_context=False,
              name=None):
    """Utility function to apply conv + BN.

    Args:
      x: input tensor.
      filters: filters in `Conv2D`.
      kernel_size: kernel size as in `Conv2D`.
      strides: strides in `Conv2D`.
      padding: padding mode in `Conv2D`.
      activation: activation in `Conv2D`.
      use_bias: whether to use a bias in `Conv2D`.
      name: name of the ops; will become `name + '_ac'` for the activation
          and `name + '_bn'` for the batch norm layer.

    Returns:
      Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if include_context == True:
        input_shape = backend.int_shape(x)
        x = GCBlock3D(in_channel=input_shape[-1])(x)
    x = layers.Conv3D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name)(
            x)
    if not use_bias:
        bn_axis = -1
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        if activation == 'relu':
            x = layers.Activation(tf.nn.relu6, name=ac_name)(x)
        else:
            x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block_3d(x, scale, block_type, block_idx, activation='relu', include_context=False):
    """Adds an Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
    - Inception-ResNet-A: `block_type='block35'`
    - Inception-ResNet-B: `block_type='block17'`
    - Inception-ResNet-C: `block_type='block8'`

    Args:
      x: input tensor.
      scale: scaling factor to scale the residuals (i.e., the output of passing
        `x` through an inception module) before adding them to the shortcut
        branch. Let `r` be the output from the residual branch, the output of this
        block will be `x + scale * r`.
      block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
        structure in the residual branch.
      block_idx: an `int` used for generating layer names. The Inception-ResNet
        blocks are repeated many times in this network. We use `block_idx` to
        identify each of the repetitions. For example, the first
        Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
        and the layer names will have a common prefix `'block35_0'`.
      activation: activation function to use at the end of the block (see
        [activations](../activations.md)). When `activation=None`, no activation
        is applied
        (i.e., "linear" activation: `a(x) = x`).

    Returns:
        Output tensor for the block.

    Raises:
      ValueError: if `block_type` is not one of `'block35'`,
        `'block17'` or `'block8'`.
    """
    if block_type == 'block35_3d':
        branch_0 = conv3d_bn(x, 160, 1, include_context=include_context)
        branch_1 = conv3d_bn(x, 160, 1, include_context=include_context)
        branch_1 = conv3d_bn(branch_1, 160, 3)
        branch_2 = conv3d_bn(x, 160, 1, include_context=include_context)
        branch_2 = conv3d_bn(branch_2, 240, 3)
        branch_2 = conv3d_bn(branch_2, 320, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17_3d':
        branch_0 = conv3d_bn(x, 288, 1, include_context=include_context)
        branch_1 = conv3d_bn(x, 192, 1, include_context=include_context)
        branch_1 = conv3d_bn(branch_1, 240, [1, 1, 7])
        branch_1 = conv3d_bn(branch_1, 264, [1, 7, 1])
        branch_1 = conv3d_bn(branch_1, 288, [7, 1, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8_3d':
        branch_0 = conv3d_bn(x, 192, 1, include_context=include_context)
        branch_1 = conv3d_bn(x, 192, 1, include_context=include_context)
        branch_1 = conv3d_bn(branch_1, 224, [1, 1, 3])
        branch_1 = conv3d_bn(branch_1, 240, [1, 3, 1])
        branch_1 = conv3d_bn(branch_1, 256, [3, 1, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35_3d", "block17_3d" or "branch_1", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = -1
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(
            branches)
    up = conv3d_bn(
        mixed,
        backend.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        include_context=include_context,
        name=block_name + '_conv')

    x = layers.Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=backend.int_shape(x)[1:],
        arguments={'scale': scale},
        name=block_name)([x, up])
    if activation is not None:
        if activation == 'relu':
            x = layers.Activation(tf.nn.relu6, name=block_name + '_ac')(x)
        else:
            x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


class SkipUpsample3D(layers.Layer):
    def __init__(self, filters, include_context=False):
        super().__init__()
        self.include_context = include_context
        self.compress_block = Sequential([
            layers.Conv2D(filters, kernel_size=1, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.BatchNormalization(axis=-1),
            layers.Activation("tanh")
        ])
        self.conv_block = Sequential([
            layers.Conv3D(filters, kernel_size=3, padding="same",
                          strides=1, use_bias=USE_CONV_BIAS),
            layers.BatchNormalization(axis=-1),
            layers.Activation("tanh")
        ])

    def call(self, input_tensor, H):
        input_shape = backend.int_shape(input_tensor)
        if self.include_context == True:
            input_tensor = GCBlock2D(in_channel=input_shape[-1])(input_tensor)
        conv = self.compress_block(input_tensor)
        # shape: [B H W 1 C]
        conv = backend.expand_dims(conv, axis=-2)
        conv = backend.repeat_elements(conv, rep=H, axis=-2)
        conv = self.conv_block(conv)
        return conv


class HighwayMulti(layers.Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, dim, activation='relu', transform_gate_bias=-3, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dim = dim
        self.dense_1 = layers.Dense(units=self.dim,
                                    use_bias=USE_DENSE_BIAS, bias_initializer=transform_gate_bias_initializer)

        super(HighwayMulti, self).__init__(**kwargs)

    def call(self, x, y):

        transform_gate = layers.GlobalAveragePooling3D()(x)
        transform_gate = self.dense_1(transform_gate)
        transform_gate = layers.Activation("sigmoid")(transform_gate)
        carry_gate = layers.Lambda(lambda x: 1.0 - x,
                                   output_shape=(self.dim,))(transform_gate)
        transformed_gated = layers.Multiply()([transform_gate, x])
        identity_gated = layers.Multiply()([carry_gate, y])
        value = layers.Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(HighwayMulti, self).get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config


class HighwayResnetDecoder3D(layers.Layer):
    def __init__(self, filters, strides):
        super().__init__()

        self.filters = filters
        self.conv_before_trans = layers.Conv3D(filters=filters,
                                               kernel_size=1, padding="same",
                                               strides=1, use_bias=USE_CONV_BIAS)
        self.conv_trans = layers.Conv3DTranspose(filters=filters,
                                                 kernel_size=3, padding="same",
                                                 strides=strides, use_bias=USE_CONV_BIAS)
        self.conv_after_trans = layers.Conv3D(filters=filters,
                                              kernel_size=1, padding="same",
                                              strides=1, use_bias=USE_CONV_BIAS)

        self.conv_before_upsample = layers.Conv3D(filters=filters,
                                                  kernel_size=1, padding="same",
                                                  strides=1, use_bias=USE_CONV_BIAS)
        self.upsample_layer = layers.UpSampling3D(size=strides)
        self.conv_after_upsample = layers.Conv3D(filters=filters,
                                                 kernel_size=1, padding="same",
                                                 strides=1, use_bias=USE_CONV_BIAS)

        self.norm_layer = layers.LayerNormalization(axis=-1)
        self.act_layer = tanh
        self.highway_layer = HighwayMulti(dim=filters)

    def call(self, input_tensor):

        conv_trans = self.conv_before_trans(input_tensor)
        conv_trans = self.conv_trans(conv_trans)
        conv_trans = self.conv_after_trans(conv_trans)

        upsamle = self.conv_before_upsample(input_tensor)
        upsamle = self.upsample_layer(upsamle)
        upsamle = self.conv_after_upsample(upsamle)

        output = self.highway_layer(conv_trans, upsamle)
        output = self.norm_layer(output)
        output = self.act_layer(output)
        return output


class OutputLayer(layers.Layer):
    def __init__(self, last_channel_num, act="tanh", use_highway=True):
        super().__init__()
        self.use_highway = use_highway

        self.conv_1x1 = layers.Conv3D(filters=last_channel_num,
                                      kernel_size=1,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        self.conv_3x3 = layers.Conv3D(filters=last_channel_num,
                                      kernel_size=3,
                                      padding="same",
                                      strides=1,
                                      use_bias=USE_CONV_BIAS,
                                      )
        # self.conv_1x1x3 = layers.Conv3D(filters=last_channel_num,
        #                                 kernel_size=(1, 1, 3),
        #                                 padding="same",
        #                                 strides=1,
        #                                 use_bias=USE_CONV_BIAS,
        #                                 )
        # self.conv_1x3x1 = layers.Conv3D(filters=last_channel_num,
        #                                 kernel_size=(1, 3, 1),
        #                                 padding="same",
        #                                 strides=1,
        #                                 use_bias=USE_CONV_BIAS,
        #                                 )
        # self.conv_3x1x1 = layers.Conv3D(filters=last_channel_num,
        #                                 kernel_size=(3, 1, 1),
        #                                 padding="same",
        #                                 strides=1,
        #                                 use_bias=USE_CONV_BIAS,
        #                                 )

        self.act = layers.Activation(act)

    def call(self, input_tensor):
        conv_1x1 = self.conv_1x1(input_tensor)
        conv_3x3 = self.conv_3x3(input_tensor)
        # conv_1x1x3 = self.conv_1x1x3(input_tensor)
        # conv_1x3x1 = self.conv_1x3x1(input_tensor)
        # conv_3x1x1 = self.conv_3x1x1(input_tensor)
        output = conv_1x1 + conv_3x3
        # output = conv_1x1 + conv_3x3 + conv_1x1x3 + conv_1x3x1 + conv_3x1x1
        output = self.act(output)

        return output
