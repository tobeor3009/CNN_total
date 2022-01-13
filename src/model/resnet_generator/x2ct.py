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

    ct_start_channel = ct_series_shape[-1] // (2 ** (encoder_depth - 1))

    decoder_depth = encoder_depth
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(1, encoder_depth + 1):
        encode_layer_1 = HighwayResnetBlock(
            init_filters * encode_i, use_highway=False)
        encode_layer_3 = HighwayResnetEncoder(init_filters * encode_i)
        setattr(layer_archive, f"encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"encode_{encode_i}_2", encode_layer_3)

    middle_layers = []
    for middle_index in range(1, middle_depth + 1):
        middle_layer = HighwayResnetBlock3D(
            init_filters * encoder_depth)
        middle_layers.append(middle_layer)

    middle_layers = Sequential(middle_layers)

    for decode_i in range(decoder_depth, 1, -1):

        decode_layer_1 = HighwayResnetBlock3D(
            init_filters * decode_i, use_highway=False)
        decode_layer_2 = HighwayResnetDecoder3D(
            init_filters * decode_i)
        setattr(layer_archive, f"decode_{decode_i}_1", decode_layer_1)
        setattr(layer_archive, f"decode_{decode_i}_2", decode_layer_2)

    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    xray_input = layers.Input(shape=xray_shape)
    encoded_tensor = HighwayResnetBlock(ct_start_channel,
                                        use_highway=False)(xray_input)

    for encode_i in range(1, encoder_depth + 1):
        encode_layer_1 = getattr(layer_archive, f"encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"encode_{encode_i}_2")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)

        if skip_connection is True:
            setattr(tensor_archive, f"encode_{encode_i}", encoded_tensor)

    encoded_tensor = HighwayResnetBlock(ct_start_channel,
                                        use_highway=False)(encoded_tensor)
    decoded_tensor = keras_backend.expand_dims(encoded_tensor, axis=-1)
    decoded_tensor = middle_layers(decoded_tensor)

    for decode_i in range(decoder_depth, 1, -1):
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")

        decoded_tensor = decode_layer_1(decoded_tensor)

        if skip_connection is True:
            skip_connection_target = getattr(
                tensor_archive, f"encode_{decode_i}")
            skip_connection_target = keras_backend.expand_dims(skip_connection_target,
                                                               axis=-2)
            decoded_tensor = decoded_tensor + skip_connection_target

        decoded_tensor = decode_layer_2(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock3D(
        init_filters, use_highway=False)(decoded_tensor)
    last_modified_tensor = HighwayResnetBlock3D(
        1, use_highway=False)(decoded_tensor)
    last_modified_tensor = layers.Activation(
        last_channel_activation)(last_modified_tensor)
    last_modified_tensor = keras_backend.squeeze(last_modified_tensor, axis=-1)

    return Model(xray_input, last_modified_tensor)
