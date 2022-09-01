import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras import layers
from .layers import LayerArchive, TensorArchive, HighwayResnetBlock, HighwayResnetEncoder
from .layers_3d import HighwayResnetBlock3D, HighwayResnetDecoder3D, base_act, SkipUpsample3D, HighwayOutputLayer, DenseBlock2D


class DenseBlock(layers.Layer):
    def __init__(self, dim, dropout_p=0.5):
        super().__init__()
        self.dense = layers.Dense(dim, use_bias=False)
        self.dropout = layers.Dropout(dropout_p)
        self.act = base_act

    def call(self, input):
        dense = self.dense(input)
        dense = self.dropout(dense)
        dense = self.act(dense)

        return dense


def get_x2ct_model(xray_shape, ct_series_shape,
                   init_filters, encoder_depth, middle_depth,
                   last_channel_activation="tanh"):
    USE_HIGH_WAY = False

    layer_archive = LayerArchive()
    tensor_archive = TensorArchive()

    ct_start_channel = ct_series_shape[-1] // (2 ** (encoder_depth - 2))

    decoder_depth = encoder_depth
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(0, encoder_depth):
        filters = init_filters * (1 + encode_i)
        encode_layer_1 = HighwayResnetBlock(filters,
                                            use_highway=USE_HIGH_WAY)
        encode_layer_2 = HighwayResnetBlock(filters,
                                            use_highway=USE_HIGH_WAY)
        encode_layer_3 = HighwayResnetEncoder(filters,
                                              use_highway=USE_HIGH_WAY)

        setattr(layer_archive, f"encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"encode_{encode_i}_2", encode_layer_2)
        setattr(layer_archive, f"encode_{encode_i}_3", encode_layer_3)

    middle_layers = []
    for _ in range(0, middle_depth):
        middle_layer = HighwayResnetBlock3D(filters,
                                            use_highway=USE_HIGH_WAY)
        middle_layers.append(middle_layer)
    middle_layers = Sequential(middle_layers)

    for decode_i in range(decoder_depth - 1, 1, -1):
        filters = init_filters * (1 + decode_i)
        decode_layer_1 = HighwayResnetBlock3D(filters,
                                              use_highway=False)
        decode_layer_2 = HighwayResnetBlock3D(filters,
                                              use_highway=False)
        decode_layer_3 = HighwayResnetDecoder3D(filters,
                                                strides=(2, 2, 2))
        setattr(layer_archive, f"decode_{decode_i}_1", decode_layer_1)
        setattr(layer_archive, f"decode_{decode_i}_2", decode_layer_2)
        setattr(layer_archive, f"decode_{decode_i}_3", decode_layer_3)
    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    xray_input = layers.Input(shape=xray_shape)
    encoded_tensor = HighwayResnetBlock(init_filters,
                                        use_highway=False)(xray_input)
    ##############################################################
    ######################### Encoder ############################
    ##############################################################
    for encode_i in range(0, encoder_depth):
        encode_layer_1 = getattr(layer_archive, f"encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"encode_{encode_i}_2")
        encode_layer_3 = getattr(layer_archive, f"encode_{encode_i}_3")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)
        encoded_tensor = encode_layer_3(encoded_tensor)

        setattr(tensor_archive, f"encode_{encode_i}", encoded_tensor)

    ##############################################################
    ######################### Middle layer #######################
    ##############################################################
    decoded_tensor = SkipUpsample3D(init_filters * (1 + encode_i)
                                    )(encoded_tensor, ct_start_channel)
    decoded_tensor = middle_layers(decoded_tensor)

    ##############################################################
    ######################### Decoder ############################
    ##############################################################
    ct_dim = ct_start_channel
    for decode_i in range(decoder_depth - 1, 1, -1):
        filters = init_filters * (1 + decode_i)
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")
        decode_layer_3 = getattr(layer_archive, f"decode_{decode_i}_3")

        skip_connect = getattr(tensor_archive, f"encode_{decode_i}")
        skip_connect = SkipUpsample3D(filters)(skip_connect, ct_dim)

        decoded_tensor = decode_layer_1(decoded_tensor)
        decoded_tensor = decode_layer_2(decoded_tensor)
        decoded_tensor = layers.Concatenate()(
            [decoded_tensor, skip_connect])

        decoded_tensor = decode_layer_3(decoded_tensor)
        ct_dim *= 2
    last_modified_tensor = HighwayResnetBlock3D(init_filters,
                                                use_highway=False)(decoded_tensor)
    last_modified_tensor = HighwayOutputLayer(last_channel_num=1,
                                              act=last_channel_activation)(last_modified_tensor)
    last_modified_tensor = keras_backend.squeeze(last_modified_tensor, axis=-1)

    return Model(xray_input, last_modified_tensor)
