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

    ct_start_channel = ct_series_shape[-1] // (2 ** (encoder_depth - 1))

    decoder_depth = encoder_depth
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(0, encoder_depth):
        filters = init_filters * (1 + encode_i)
        ap_encode_layer_1 = HighwayResnetBlock(filters,
                                               use_highway=False)
        ap_encode_layer_2 = DenseBlock2D(filters)
        lat_encode_layer_1 = HighwayResnetBlock(filters,
                                                use_highway=False)
        lat_encode_layer_2 = DenseBlock2D(filters)

        setattr(layer_archive, f"ap_encode_{encode_i}_1", ap_encode_layer_1)
        setattr(layer_archive, f"ap_encode_{encode_i}_2", ap_encode_layer_2)
        setattr(layer_archive, f"lat_encode_{encode_i}_1", lat_encode_layer_1)
        setattr(layer_archive, f"lat_encode_{encode_i}_2", lat_encode_layer_2)

    ap_middle_layers = []
    lat_middle_layers = []
    concat_middle_layers = []
    for _ in range(0, middle_depth):
        ap_middle_layer = HighwayResnetBlock3D(
            filters, use_highway=USE_HIGH_WAY)
        lat_middle_layer = HighwayResnetBlock3D(
            filters, use_highway=USE_HIGH_WAY)
        concat_middle_layer = HighwayResnetBlock3D(
            filters, use_highway=USE_HIGH_WAY)

        ap_middle_layers.append(ap_middle_layer)
        lat_middle_layers.append(lat_middle_layer)
        concat_middle_layers.append(concat_middle_layer)
    ap_middle_layers = Sequential(ap_middle_layers)
    lat_middle_layers = Sequential(lat_middle_layers)
    concat_middle_layers = Sequential(concat_middle_layers)

    for decode_i in range(decoder_depth - 1, 0, -1):
        filters = init_filters * (1 + decode_i)
        ap_decode_layer_1 = HighwayResnetBlock3D(filters,
                                                 use_highway=False)
        ap_decode_layer_2 = HighwayResnetDecoder3D(filters,
                                                   strides=(2, 2, 2))
        lat_decode_layer_1 = HighwayResnetBlock3D(filters,
                                                  use_highway=False)
        lat_decode_layer_2 = HighwayResnetDecoder3D(filters,
                                                    strides=(2, 2, 2))
        concat_decode_layer_1 = HighwayResnetBlock3D(filters,
                                                     use_highway=False)
        concat_decode_layer_2 = HighwayResnetDecoder3D(filters,
                                                       strides=(2, 2, 2))
        concat_decode_layer_3 = HighwayResnetBlock3D(filters,
                                                     use_highway=False)
        setattr(layer_archive, f"ap_decode_{decode_i}_1", ap_decode_layer_1)
        setattr(layer_archive, f"ap_decode_{decode_i}_2", ap_decode_layer_2)
        setattr(layer_archive, f"lat_decode_{decode_i}_1", lat_decode_layer_1)
        setattr(layer_archive, f"lat_decode_{decode_i}_2", lat_decode_layer_2)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_1", concat_decode_layer_1)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_2", concat_decode_layer_2)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_3", concat_decode_layer_3)
    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    xray_input = layers.Input(shape=xray_shape)
    ap_encoded_tensor = HighwayResnetBlock(init_filters,
                                           use_highway=False)(xray_input[..., :1])
    lat_encoded_tensor = HighwayResnetBlock(init_filters,
                                            use_highway=False)(xray_input[..., 1:2])
    ##############################################################
    ######################### Encoder ############################
    ##############################################################
    for encode_i in range(0, encoder_depth):
        ap_encode_layer_1 = getattr(layer_archive, f"ap_encode_{encode_i}_1")
        ap_encode_layer_2 = getattr(layer_archive, f"ap_encode_{encode_i}_2")
        lat_encode_layer_1 = getattr(layer_archive, f"lat_encode_{encode_i}_1")
        lat_encode_layer_2 = getattr(layer_archive, f"lat_encode_{encode_i}_2")

        ap_encoded_tensor = ap_encode_layer_1(ap_encoded_tensor)
        ap_encoded_tensor = ap_encode_layer_2(ap_encoded_tensor)
        lat_encoded_tensor = lat_encode_layer_1(lat_encoded_tensor)
        lat_encoded_tensor = lat_encode_layer_2(lat_encoded_tensor)

        setattr(tensor_archive, f"ap_encode_{encode_i}", ap_encoded_tensor)
        setattr(tensor_archive, f"lat_encode_{encode_i}", lat_encoded_tensor)

    concat_encoded_tensor = layers.Concatenate()(
        [ap_encoded_tensor, lat_encoded_tensor])
    ##############################################################
    ######################### Middle layer #######################
    ##############################################################
    ap_decoded_tensor = SkipUpsample3D(init_filters * (1 + encode_i)
                                       )(ap_encoded_tensor, ct_start_channel)
    lat_decoded_tensor = SkipUpsample3D(init_filters * (1 + encode_i)
                                        )(lat_encoded_tensor, ct_start_channel)
    concat_decoded_tensor = SkipUpsample3D(init_filters * (1 + encode_i)
                                           )(concat_encoded_tensor, ct_start_channel)
    ap_decoded_tensor = ap_middle_layers(ap_decoded_tensor)
    lat_decoded_tensor = lat_middle_layers(lat_decoded_tensor)
    concat_decoded_tensor = concat_middle_layers(concat_decoded_tensor)

    ##############################################################
    ######################### Decoder ############################
    ##############################################################
    ct_dim = ct_start_channel
    for decode_i in range(decoder_depth - 1, 0, -1):
        filters = init_filters * (1 + decode_i)
        ap_decode_layer_1 = getattr(layer_archive, f"ap_decode_{decode_i}_1")
        ap_decode_layer_2 = getattr(layer_archive, f"ap_decode_{decode_i}_2")
        lat_decode_layer_1 = getattr(layer_archive, f"lat_decode_{decode_i}_1")
        lat_decode_layer_2 = getattr(layer_archive, f"lat_decode_{decode_i}_2")
        concat_decode_layer_1 = getattr(
            layer_archive, f"concat_decode_{decode_i}_1")
        concat_decode_layer_2 = getattr(
            layer_archive, f"concat_decode_{decode_i}_2")
        concat_decode_layer_3 = getattr(
            layer_archive, f"concat_decode_{decode_i}_3")
        ap_decoded_tensor = ap_decode_layer_1(ap_decoded_tensor)
        lat_decoded_tensor = ap_decode_layer_1(lat_decoded_tensor)

        ap_skip_connect = getattr(tensor_archive, f"ap_encode_{decode_i}")
        lat_skip_connect = getattr(tensor_archive, f"lat_encode_{decode_i}")

        ap_skip_connect = SkipUpsample3D(filters)(ap_skip_connect, ct_dim)
        lat_skip_connect = SkipUpsample3D(filters)(lat_skip_connect, ct_dim)

        ap_decoded_tensor = layers.Concatenate()(
            [ap_decoded_tensor, ap_skip_connect])
        lat_decoded_tensor = layers.Concatenate()(
            [lat_decoded_tensor, lat_skip_connect])

        ap_decoded_tensor = ap_decode_layer_2(ap_decoded_tensor)
        lat_decoded_tensor = lat_decode_layer_2(lat_decoded_tensor)

        concat_decoded_tensor = concat_decode_layer_1(concat_decoded_tensor)
        concat_decoded_tensor = concat_decode_layer_2(concat_decoded_tensor)
        concat_decoded_tensor = layers.Concatenate()(
            [concat_decoded_tensor, ap_decoded_tensor, lat_decoded_tensor])

        concat_decoded_tensor = concat_decode_layer_3(concat_decoded_tensor)
        ct_dim *= 2
    last_modified_tensor = HighwayResnetBlock3D(
        init_filters, use_highway=False)(concat_decoded_tensor)
    last_modified_tensor = HighwayOutputLayer(last_channel_num=1,
                                              act=last_channel_activation)(last_modified_tensor)
    last_modified_tensor = keras_backend.squeeze(last_modified_tensor, axis=-1)

    return Model(xray_input, last_modified_tensor)
