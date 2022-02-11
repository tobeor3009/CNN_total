from numpy import expand_dims
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, Sequential, Model
from .layers import LayerArchive, TensorArchive, HighwayResnetBlock, HighwayResnetEncoder
from .layers_3d import HighwayResnetBlock3D, HighwayResnetDecoder3D


def get_x2ct_model(xray_shape, ct_series_shape,
                   init_filters, encoder_depth, middle_depth,
                   last_channel_activation="tanh", skip_connection=True
                   ):
    layer_archive = LayerArchive()
    tensor_archive = TensorArchive()

    ct_start_channel = ct_series_shape[-1] // (1 + (encoder_depth - 1))

    decoder_depth = encoder_depth
    middle_decoder_depth = middle_decoder_depth - 3
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(0, encoder_depth):
        encode_layer_1 = HighwayResnetBlock(
            init_filters * (1 + encode_i), use_highway=False)
        encode_layer_2 = HighwayResnetEncoder(init_filters * (1 + encode_i))
        setattr(layer_archive, f"encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"encode_{encode_i}_2", encode_layer_2)

    middle_layers = []
    for _ in range(0, middle_depth):
        middle_layer = HighwayResnetBlock3D(
            init_filters * (1 + encode_i))
        middle_layers.append(middle_layer)
    middle_layers = Sequential(middle_layers)

    middle_decoder_layers = []
    for _ in range(0, middle_decoder_depth):
        middle_layer_1 = HighwayResnetBlock3D(
            init_filters * (1 + encode_i))
        middle_layer_2 = HighwayResnetDecoder3D(
            init_filters * (1 + encode_i), strides=(1, 1, 2))
        middle_decoder_layers.append(middle_layer_1)
        middle_decoder_layers.append(middle_layer_2)
    middle_decoder_layers = Sequential(middle_decoder_layers)
    for decode_i in range(decoder_depth - 1, 0, -1):

        decode_layer_1 = HighwayResnetBlock3D(
            init_filters * (1 + decode_i), use_highway=False)
        decode_layer_2 = HighwayResnetDecoder3D(
            init_filters * (1 + decode_i), strides=(1, 1, 2))
        decode_layer_3 = HighwayResnetBlock3D(
            init_filters * (1 + decode_i), use_highway=False)
        decode_layer_4 = HighwayResnetDecoder3D(
            init_filters * (1 + decode_i), strides=(2, 2, 2))

        setattr(layer_archive, f"decode_{decode_i}_1", decode_layer_1)
        setattr(layer_archive, f"decode_{decode_i}_2", decode_layer_2)
        setattr(layer_archive, f"decode_{decode_i}_3", decode_layer_3)
        setattr(layer_archive, f"decode_{decode_i}_4", decode_layer_4)

    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    xray_input = layers.Input(shape=xray_shape)
    encoded_tensor = HighwayResnetBlock(ct_start_channel,
                                        use_highway=False)(xray_input)

    for encode_i in range(0, encoder_depth):
        encode_layer_1 = getattr(layer_archive, f"encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"encode_{encode_i}_2")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)

        if skip_connection is True:
            setattr(tensor_archive, f"encode_{encode_i}", encoded_tensor)

    encoded_tensor = encoded_tensor.expand_dims(encoded_tensor, axis=-1)
    decoded_tensor = middle_layers(encoded_tensor)
    decoded_tensor = middle_decoder_layers(encoded_tensor)

    for decode_i in range(decoder_depth - 1, 0, -1):
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")
        decode_layer_3 = getattr(layer_archive, f"decode_{decode_i}_3")
        decode_layer_4 = getattr(layer_archive, f"decode_{decode_i}_4")

        decoded_tensor = decode_layer_1(decoded_tensor)
        decoded_tensor = decode_layer_2(decoded_tensor)
        decoded_tensor = decode_layer_3(decoded_tensor)

        if skip_connection is True:
            skip_connection_target = getattr(
                tensor_archive, f"encode_{decode_i}")
            skip_connection_target = keras_backend.expand_dims(skip_connection_target,
                                                               axis=-2)
            decoded_tensor = decoded_tensor + skip_connection_target

        decoded_tensor = decode_layer_4(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock3D(
        init_filters, use_highway=False)(decoded_tensor)
    last_modified_tensor = layers.Conv3D(filters=1, kernel_size=1,
                                         padding="same", strides=1,
                                         use_bias=False)(last_modified_tensor)
    last_modified_tensor = layers.Activation(
        last_channel_activation)(last_modified_tensor)
    last_modified_tensor = keras_backend.squeeze(last_modified_tensor, axis=-1)

    return Model(xray_input, last_modified_tensor)
