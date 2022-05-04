import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras import layers
from .layers import LayerArchive, TensorArchive, HighwayResnetBlock, HighwayResnetEncoder
from .layers_3d import HighwayResnetBlock3D, HighwayResnetDecoder3D, base_act, SkipUpsample3D


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
                   last_channel_activation="tanh", skip_connection=True
                   ):
    layer_archive = LayerArchive()
    tensor_archive = TensorArchive()

    ct_start_channel = ct_series_shape[-1] // (2 ** (encoder_depth - 1))

    decoder_depth = encoder_depth
    middle_decoder_depth = 5 - encoder_depth
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    ct_series_shape = np.array([xray_shape[0], xray_shape[0], xray_shape[0]])

    for encode_i in range(0, encoder_depth):
        if encode_i == 0:
            in_channel = init_filters
        else:
            in_channel = init_filters * (2 ** (encode_i - 1))
        encode_layer_1 = HighwayResnetBlock(out_channel=init_filters * (2 ** encode_i),
                                            in_channel=in_channel,
                                            use_highway=False)
        encode_layer_2 = HighwayResnetBlock(init_filters * (2 ** encode_i))
        encode_layer_3 = HighwayResnetEncoder(init_filters * (2 ** encode_i),
                                              include_context=True)
        setattr(layer_archive, f"encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"encode_{encode_i}_2", encode_layer_2)
        setattr(layer_archive, f"encode_{encode_i}_3", encode_layer_3)

    for encode_i in range(0, encoder_depth):
        ct_series_shape //= 2
        channel_num = init_filters * (2 ** encode_i)

        encode_layer_1 = DenseBlock(channel_num * 4)
        encode_layer_2 = DenseBlock(channel_num * 2)
        encode_layer_3 = DenseBlock(channel_num, 0)
        setattr(layer_archive, f"dense_encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"dense_encode_{encode_i}_2", encode_layer_2)
        setattr(layer_archive, f"dense_encode_{encode_i}_3", encode_layer_3)

    ct_series_element_num = np.prod(ct_series_shape)

    middle_decode_layers = []
    middle_decode_layers.append(DenseBlock(ct_series_element_num * 3))
    middle_decode_layers.append(layers.Reshape((*ct_series_shape, 3)))
    middle_decode_layers = Sequential(middle_decode_layers)

    middle_layers = []
    for _ in range(0, middle_depth):
        middle_layer = HighwayResnetBlock3D(channel_num, use_highway=False)
        middle_layers.append(middle_layer)
    middle_layers = Sequential(middle_layers)

    for decode_i in range(decoder_depth - 1, 0, -1):
        decode_layer_1 = HighwayResnetBlock3D(
            init_filters * (2 ** decode_i), use_highway=False)
        decode_layer_2 = HighwayResnetDecoder3D(
            init_filters * (2 ** decode_i), strides=(2, 2, 2))

        setattr(layer_archive, f"decode_{decode_i}_1", decode_layer_1)
        setattr(layer_archive, f"decode_{decode_i}_2", decode_layer_2)

    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    xray_input = layers.Input(shape=xray_shape)
    encoded_tensor = HighwayResnetBlock(init_filters,
                                        use_highway=False)(xray_input)
    for encode_i in range(0, encoder_depth):
        encode_layer_1 = getattr(layer_archive, f"encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"encode_{encode_i}_2")
        encode_layer_3 = getattr(layer_archive, f"encode_{encode_i}_3")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)
        encoded_tensor = encode_layer_3(encoded_tensor)

    encoded_tensor = layers.GlobalAveragePooling2D()(encoded_tensor)
    encoded_tensor = DenseBlock(channel_num * 2)(encoded_tensor)

    for encode_i in range(0, encoder_depth):
        encode_layer_1 = getattr(layer_archive, f"dense_encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"dense_encode_{encode_i}_2")
        encode_layer_3 = getattr(layer_archive, f"dense_encode_{encode_i}_3")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)
        encoded_tensor = encode_layer_3(encoded_tensor)

        if skip_connection is True:
            setattr(tensor_archive, f"encode_{encode_i}", encoded_tensor)

    decoded_tensor = middle_decode_layers(encoded_tensor)
    decoded_tensor = middle_layers(decoded_tensor)

    for decode_i in range(decoder_depth - 1, 0, -1):
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")

        decoded_tensor = decode_layer_1(decoded_tensor)

        if skip_connection is True:
            channel_num = init_filters * (2 ** decode_i)
            skip_connection_target = getattr(
                tensor_archive, f"encode_{decode_i}")
            skip_connection_target = layers.Reshape(
                (1, 1, 1, channel_num))(skip_connection_target)
            print(skip_connection_target.shape)
            print(decoded_tensor.shape)
            decoded_tensor = layers.Add()(
                [decoded_tensor, skip_connection_target])
            ct_series_shape *= 2
        decoded_tensor = decode_layer_2(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock3D(
        init_filters, use_highway=False)(decoded_tensor)
    last_modified_tensor = layers.Conv3D(filters=1, kernel_size=3,
                                         padding="same", strides=1,
                                         use_bias=False)(last_modified_tensor)
    last_modified_tensor = layers.Activation(
        last_channel_activation)(last_modified_tensor)
    last_modified_tensor = keras_backend.squeeze(last_modified_tensor, axis=-1)

    return Model(xray_input, last_modified_tensor)


def get_x2ct_model(xray_shape, ct_series_shape,
                   init_filters, encoder_depth, middle_depth,
                   last_channel_activation="tanh", skip_connection=True
                   ):
    layer_archive = LayerArchive()
    tensor_archive = TensorArchive()

    ct_start_channel = ct_series_shape[-1] // (2 ** (encoder_depth - 1))

    decoder_depth = encoder_depth
    middle_decoder_depth = 5 - encoder_depth
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(0, encoder_depth):
        filters = init_filters * (2 ** encode_i)
        ap_encode_layer_1 = HighwayResnetBlock(filters,
                                               use_highway=False)
        ap_encode_layer_2 = HighwayResnetEncoder(filters)
        lat_encode_layer_1 = HighwayResnetBlock(filters,
                                                use_highway=False)
        lat_encode_layer_2 = HighwayResnetEncoder(filters)

        setattr(layer_archive, f"ap_encode_{encode_i}_1", ap_encode_layer_1)
        setattr(layer_archive, f"ap_encode_{encode_i}_2", ap_encode_layer_2)
        setattr(layer_archive, f"lat_encode_{encode_i}_1", lat_encode_layer_1)
        setattr(layer_archive, f"lat_encode_{encode_i}_2", lat_encode_layer_2)

    ap_middle_decoder_layers = []
    lat_middle_decoder_layers = []
    concat_middle_decoder_layers = []

    for _ in range(0, middle_decoder_depth):
        ap_middle_layer_1 = HighwayResnetBlock3D(filters)
        ap_middle_layer_2 = HighwayResnetDecoder3D(filters,
                                                   strides=(1, 1, 2))
        lat_middle_layer_1 = HighwayResnetBlock3D(filters)
        lat_middle_layer_2 = HighwayResnetDecoder3D(filters,
                                                    strides=(1, 1, 2))
        concat_middle_layer_1 = HighwayResnetBlock3D(filters)
        concat_middle_layer_2 = HighwayResnetDecoder3D(filters,
                                                       strides=(1, 1, 2))
        ap_middle_decoder_layers.append(ap_middle_layer_1)
        ap_middle_decoder_layers.append(ap_middle_layer_2)
        lat_middle_decoder_layers.append(lat_middle_layer_1)
        lat_middle_decoder_layers.append(lat_middle_layer_2)
        concat_middle_decoder_layers.append(concat_middle_layer_1)
        concat_middle_decoder_layers.append(concat_middle_layer_2)

    ap_middle_decoder_layers = Sequential(ap_middle_decoder_layers)
    lat_middle_decoder_layers = Sequential(lat_middle_decoder_layers)
    concat_middle_decoder_layers = Sequential(concat_middle_decoder_layers)

    ap_middle_layers = []
    lat_middle_layers = []
    concat_middle_layers = []
    for _ in range(0, middle_depth):
        ap_middle_layer = HighwayResnetBlock3D(filters)
        lat_middle_layer = HighwayResnetBlock3D(filters)
        concat_middle_layer = HighwayResnetBlock3D(filters)

        ap_middle_layers.append(ap_middle_layer)
        lat_middle_layers.append(lat_middle_layer)
        concat_middle_layers.append(concat_middle_layer)
    ap_middle_layers = Sequential(ap_middle_layers)
    lat_middle_layers = Sequential(lat_middle_layers)
    concat_middle_layers = Sequential(concat_middle_layers)

    for decode_i in range(decoder_depth - 1, 0, -1):
        filters = init_filters * (2 ** decode_i)
        ap_decode_layer_1 = HighwayResnetBlock3D(filters,
                                                 use_highway=False)
        ap_decode_layer_2 = HighwayResnetDecoder3D(filters,
                                                   strides=(1, 1, 2))
        ap_decode_layer_3 = HighwayResnetBlock3D(filters, use_highway=False)
        ap_decode_layer_4 = HighwayResnetDecoder3D(filters,
                                                   strides=(2, 2, 2))
        lat_decode_layer_1 = HighwayResnetBlock3D(filters,
                                                  use_highway=False)
        lat_decode_layer_2 = HighwayResnetDecoder3D(filters,
                                                    strides=(1, 1, 2))
        lat_decode_layer_3 = HighwayResnetBlock3D(filters, use_highway=False)
        lat_decode_layer_4 = HighwayResnetDecoder3D(filters,
                                                    strides=(2, 2, 2))
        concat_decode_layer_1 = HighwayResnetBlock3D(filters,
                                                     use_highway=False)
        concat_decode_layer_2 = HighwayResnetDecoder3D(filters,
                                                       strides=(1, 1, 2))
        concat_decode_layer_3 = HighwayResnetBlock3D(
            filters, use_highway=False)
        concat_decode_layer_4 = HighwayResnetDecoder3D(filters,
                                                       strides=(2, 2, 2))

        setattr(layer_archive, f"ap_decode_{decode_i}_1", ap_decode_layer_1)
        setattr(layer_archive, f"ap_decode_{decode_i}_2", ap_decode_layer_2)
        setattr(layer_archive, f"ap_decode_{decode_i}_3", ap_decode_layer_3)
        setattr(layer_archive, f"ap_decode_{decode_i}_4", ap_decode_layer_4)
        setattr(layer_archive, f"lat_decode_{decode_i}_1", lat_decode_layer_1)
        setattr(layer_archive, f"lat_decode_{decode_i}_2", lat_decode_layer_2)
        setattr(layer_archive, f"lat_decode_{decode_i}_3", lat_decode_layer_3)
        setattr(layer_archive, f"lat_decode_{decode_i}_4", lat_decode_layer_4)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_1", concat_decode_layer_1)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_2", concat_decode_layer_2)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_3", concat_decode_layer_3)
        setattr(layer_archive,
                f"concat_decode_{decode_i}_4", concat_decode_layer_4)
    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    xray_input = layers.Input(shape=xray_shape)
    ap_encoded_tensor = HighwayResnetBlock(init_filters,
                                           use_highway=False)(xray_input)
    lat_encoded_tensor = HighwayResnetBlock(init_filters,
                                            use_highway=False)(xray_input)

    for encode_i in range(0, encoder_depth):
        ap_encode_layer_1 = getattr(layer_archive, f"ap_encode_{encode_i}_1")
        ap_encode_layer_2 = getattr(layer_archive, f"ap_encode_{encode_i}_2")
        lat_encode_layer_1 = getattr(layer_archive, f"lat_encode_{encode_i}_1")
        lat_encode_layer_2 = getattr(layer_archive, f"lat_encode_{encode_i}_2")

        ap_encoded_tensor = ap_encode_layer_1(ap_encoded_tensor)
        ap_encoded_tensor = ap_encode_layer_2(ap_encoded_tensor)
        lat_encoded_tensor = lat_encode_layer_1(lat_encoded_tensor)
        lat_encoded_tensor = lat_encode_layer_2(lat_encoded_tensor)

        if skip_connection is True:
            setattr(tensor_archive,
                    f"ap_encode_{encode_i}", ap_encoded_tensor)
            setattr(tensor_archive,
                    f"lat_encode_{encode_i}", lat_encoded_tensor)

    concat_encoded_tensor = layers.Concateate()(
        [ap_encoded_tensor, lat_encoded_tensor])

    ap_encoded_tensor = keras_backend.expand_dims(ap_encoded_tensor, axis=-2)
    ap_decoded_tensor = ap_middle_decoder_layers(ap_encoded_tensor)
    ap_decoded_tensor = ap_middle_layers(ap_decoded_tensor)

    lat_encoded_tensor = keras_backend.expand_dims(lat_encoded_tensor, axis=-2)
    lat_decoded_tensor = lat_middle_decoder_layers(lat_encoded_tensor)
    lat_decoded_tensor = lat_middle_layers(lat_decoded_tensor)

    concat_encoded_tensor = keras_backend.expand_dims(
        concat_encoded_tensor, axis=-2)
    concat_decoded_tensor = concat_middle_decoder_layers(concat_encoded_tensor)
    concat_decoded_tensor = concat_middle_layers(concat_decoded_tensor)

    ct_dim = 2 ** middle_decoder_depth

    for decode_i in range(decoder_depth - 1, 0, -1):
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")
        decode_layer_3 = getattr(layer_archive, f"decode_{decode_i}_3")
        decode_layer_4 = getattr(layer_archive, f"decode_{decode_i}_4")

        decoded_tensor = decode_layer_1(decoded_tensor)
        decoded_tensor = decode_layer_2(decoded_tensor)
        decoded_tensor = decode_layer_3(decoded_tensor)
        ct_dim *= 2
        if skip_connection is True:
            skip_connection_target = getattr(
                tensor_archive, f"encode_{decode_i}")
            skip_connection_target = HighwayResnetBlock(init_filters * (2 ** decode_i),
                                                        use_highway=True)(skip_connection_target)

            skip_connection_target = SkipUpsample3D(
                init_filters * (2 ** decode_i))(skip_connection_target, ct_dim)

            decoded_tensor = layers.Concatenate()(
                [decoded_tensor, skip_connection_target])

        decoded_tensor = decode_layer_4(decoded_tensor)
        ct_dim *= 2

    last_modified_tensor = HighwayResnetBlock3D(
        init_filters, use_highway=False)(decoded_tensor)
    last_modified_tensor = layers.Conv3D(filters=1, kernel_size=3,
                                         padding="same", strides=1,
                                         use_bias=False)(last_modified_tensor)
    last_modified_tensor = layers.Activation(
        last_channel_activation)(last_modified_tensor)
    last_modified_tensor = keras_backend.squeeze(last_modified_tensor, axis=-1)

    return Model(xray_input, last_modified_tensor)
