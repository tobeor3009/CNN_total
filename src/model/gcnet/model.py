import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, activations, Sequential, Model
from tensorflow.keras.initializers import RandomNormal

from .layers import LayerArchive, TensorArchive, HighwayResnetBlock, HighwayResnetEncoder, HighwayResnetDecoder, get_input_label2image_tensor, base_act


def get_highway_resnet_generator_unet(input_shape,
                                      init_filters, encoder_depth, middle_depth,
                                      last_channel_num, last_channel_activation="tanh",
                                      skip_connection=True):

    decoder_depth = encoder_depth

    layer_archive = LayerArchive()
    tensor_archive = TensorArchive()
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(1, encoder_depth + 1):
        if encode_i == 1:
            in_channel = init_filters * init_filters
        else:
            in_channel = init_filters * encode_i - 1
        encode_layer_1 = HighwayResnetBlock(out_channel=init_filters * encode_i,
                                            in_channel=in_channel,
                                            use_highway=False)
        encode_layer_2 = HighwayResnetBlock(init_filters * encode_i)
        encode_layer_3 = HighwayResnetEncoder(init_filters * encode_i)
        setattr(layer_archive, f"encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"encode_{encode_i}_2", encode_layer_2)
        setattr(layer_archive, f"encode_{encode_i}_3", encode_layer_3)

    middle_layers = []
    for middle_index in range(1, middle_depth + 1):
        if middle_index == 1:
            use_highway = False
        else:
            use_highway = True
        middle_layer = HighwayResnetBlock(
            init_filters * encoder_depth, use_highway=use_highway)
        middle_layers.append(middle_layer)
    middle_layers = Sequential(middle_layers)

    for decode_i in range(decoder_depth, 0, -1):

        if decode_i == decoder_depth:
            in_channel = init_filters * decoder_depth
        else:
            in_channel = init_filters * decode_i + 1

        decode_layer_1 = HighwayResnetBlock(out_channel=init_filters * decode_i,
                                            in_channel=in_channel,
                                            use_highway=False)
        decode_layer_2 = HighwayResnetBlock(init_filters * decode_i)
        decode_layer_3 = HighwayResnetDecoder(out_channel=init_filters * decode_i,
                                              unsharp=True)
        setattr(layer_archive, f"decode_{decode_i}_1", decode_layer_1)
        setattr(layer_archive, f"decode_{decode_i}_2", decode_layer_2)
        setattr(layer_archive, f"decode_{decode_i}_3", decode_layer_3)

    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    input_tensor = layers.Input(shape=input_shape)
    encoded_tensor = HighwayResnetBlock(out_channel=init_filters,
                                        in_channel=input_shape[-1],
                                        use_highway=False)(input_tensor)

    for encode_i in range(1, encoder_depth + 1):
        encode_layer_1 = getattr(layer_archive, f"encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"encode_{encode_i}_2")
        encode_layer_3 = getattr(layer_archive, f"encode_{encode_i}_3")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)
        encoded_tensor = encode_layer_3(encoded_tensor)

        if skip_connection is True:
            setattr(tensor_archive, f"encode_{encode_i}", encoded_tensor)

    decoded_tensor = middle_layers(encoded_tensor)

    for decode_i in range(decoder_depth, 0, -1):
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")
        decode_layer_3 = getattr(layer_archive, f"decode_{decode_i}_3")

        decoded_tensor = decode_layer_1(decoded_tensor)
        decoded_tensor = decode_layer_2(decoded_tensor)

        if skip_connection is True:
            skip_connection_target = getattr(
                tensor_archive, f"encode_{decode_i}")
            decoded_tensor = layers.Concatenate(
                axis=-1)([decoded_tensor, skip_connection_target])
        decoded_tensor = decode_layer_3(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock(
        init_filters, use_highway=False, include_context=False)(decoded_tensor)
    last_modified_tensor = HighwayResnetBlock(
        init_filters, use_highway=False, include_context=False)(decoded_tensor)
    last_modified_tensor = HighwayResnetBlock(
        last_channel_num, use_highway=False, include_context=False)(last_modified_tensor)
    last_modified_tensor = layers.Activation(
        last_channel_activation)(last_modified_tensor)
    return Model(input_tensor, last_modified_tensor)


def get_highway_resnet_generator_stargan_unet(input_shape, label_len, target_label_len,
                                              init_filters, encoder_depth, middle_depth,
                                              last_channel_num, last_channel_activation="tanh",
                                              skip_connection=True):

    decoder_depth = encoder_depth
    layer_archive = LayerArchive()
    tensor_archive = TensorArchive()
    input_label_shape = (input_shape[0] // (2 ** encoder_depth),
                         input_shape[1] // (2 ** encoder_depth),
                         init_filters * encoder_depth)

    target_label_shape = (input_shape[0] // (2 ** encoder_depth),
                          input_shape[1] // (2 ** encoder_depth),
                          init_filters * encoder_depth)
    class_input, class_tensor = get_input_label2image_tensor(label_len=label_len, target_shape=input_label_shape,
                                                             activation="tanh", negative_ratio=0.25,
                                                             dropout_ratio=0.5, reduce_level=5)
    target_class_input, target_class_tensor = get_input_label2image_tensor(label_len=target_label_len, target_shape=target_label_shape,
                                                                           activation="tanh", negative_ratio=0.25,
                                                                           dropout_ratio=0.5, reduce_level=5)
    ##############################################################
    ######################## Define Layer ########################
    ##############################################################
    for encode_i in range(1, encoder_depth + 1):
        if encode_i == 1:
            in_channel = init_filters * init_filters
        else:
            in_channel = init_filters * encode_i - 1
        encode_layer_1 = HighwayResnetBlock(out_channel=init_filters * encode_i,
                                            in_channel=in_channel,
                                            use_highway=False)
        encode_layer_2 = HighwayResnetBlock(init_filters * encode_i)
        encode_layer_3 = HighwayResnetEncoder(init_filters * encode_i)
        setattr(layer_archive, f"encode_{encode_i}_1", encode_layer_1)
        setattr(layer_archive, f"encode_{encode_i}_2", encode_layer_2)
        setattr(layer_archive, f"encode_{encode_i}_3", encode_layer_3)

    middle_layers = []
    for middle_index in range(1, middle_depth + 1):
        if middle_index == 1:
            use_highway = False
        else:
            use_highway = True
        middle_layer = HighwayResnetBlock(
            init_filters * encoder_depth, use_highway=use_highway)
        middle_layers.append(middle_layer)
    middle_layers = Sequential(middle_layers)

    for decode_i in range(decoder_depth, 0, -1):

        if decode_i == decoder_depth:
            in_channel = init_filters * decoder_depth
        else:
            in_channel = init_filters * decode_i + 1

        decode_layer_1 = HighwayResnetBlock(out_channel=init_filters * decode_i,
                                            in_channel=in_channel,
                                            use_highway=False)
        decode_layer_2 = HighwayResnetBlock(init_filters * decode_i)
        decode_layer_3 = HighwayResnetDecoder(out_channel=init_filters * decode_i,
                                              unsharp=True)
        setattr(layer_archive, f"decode_{decode_i}_1", decode_layer_1)
        setattr(layer_archive, f"decode_{decode_i}_2", decode_layer_2)
        setattr(layer_archive, f"decode_{decode_i}_3", decode_layer_3)
    ##############################################################
    ######################### Model Start ########################
    ##############################################################
    input_tensor = layers.Input(shape=input_shape)
    encoded_tensor = HighwayResnetBlock(out_channel=init_filters,
                                        in_channel=input_shape[-1],
                                        use_highway=False)(input_tensor)

    for encode_i in range(1, encoder_depth + 1):
        encode_layer_1 = getattr(layer_archive, f"encode_{encode_i}_1")
        encode_layer_2 = getattr(layer_archive, f"encode_{encode_i}_2")
        encode_layer_3 = getattr(layer_archive, f"encode_{encode_i}_3")

        encoded_tensor = encode_layer_1(encoded_tensor)
        encoded_tensor = encode_layer_2(encoded_tensor)
        encoded_tensor = encode_layer_3(encoded_tensor)

        if skip_connection is True:
            setattr(tensor_archive, f"encode_{encode_i}", encoded_tensor)

    encoded_tensor = layers.Concatenate(
        axis=-1)([encoded_tensor, class_tensor])

    feature_selcted_tensor = middle_layers(encoded_tensor)

    decoded_tensor = layers.Concatenate(
        axis=-1)([feature_selcted_tensor, target_class_tensor])

    for decode_i in range(decoder_depth, 0, -1):
        decode_layer_1 = getattr(layer_archive, f"decode_{decode_i}_1")
        decode_layer_2 = getattr(layer_archive, f"decode_{decode_i}_2")
        decode_layer_3 = getattr(layer_archive, f"decode_{decode_i}_3")

        decoded_tensor = decode_layer_1(decoded_tensor)
        decoded_tensor = decode_layer_2(decoded_tensor)

        if skip_connection is True:
            skip_connection_target = getattr(
                tensor_archive, f"encode_{decode_i}")
            decoded_tensor = layers.Concatenate(
                axis=-1)([decoded_tensor, skip_connection_target])
        decoded_tensor = decode_layer_3(decoded_tensor)

    last_modified_tensor = HighwayResnetBlock(
        init_filters, use_highway=False, include_context=False)(decoded_tensor)
    last_modified_tensor = HighwayResnetBlock(
        init_filters, use_highway=False, include_context=False)(decoded_tensor)
    last_modified_tensor = HighwayResnetBlock(
        last_channel_num, use_highway=False, include_context=False)(last_modified_tensor)
    return Model([input_tensor, class_input, target_class_input], last_modified_tensor)


def get_discriminator(
    input_img_shape,
    output_img_shape=None,
    depth=None,
    init_filters=32,
    num_class=None,
    include_validity=True,
    include_classifier=False
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_image = layers.Input(shape=input_img_shape)

    if output_img_shape is None:
        combined_imgs = original_image
        model_input = original_image
        init_channel = input_img_shape[-1]
    else:
        predicted_image = layers.Input(shape=output_img_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = layers.Concatenate(
            axis=-1)([original_image, predicted_image])
        model_input = [original_image, predicted_image]
        init_channel = input_img_shape[-1] + output_img_shape[-1]

    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 3
    decoded_shape = (input_img_shape[0] // (2 ** depth),
                     input_img_shape[1] // (2 ** depth),
                     init_filters * (2 ** (depth // 2)))
    decoded_element_num = np.prod(decoded_shape)
    decoded_tensor = HighwayResnetBlock(out_channel=init_filters,
                                        in_channel=init_channel,
                                        use_highway=False)(combined_imgs)

    for encode_i in range(depth):
        if encode_i == 0:
            in_channel = init_filters
        else:
            in_channel = init_filters * (2 ** (encode_i - 1))
        decoded_tensor = HighwayResnetBlock(out_channel=init_filters * (2 ** encode_i),
                                            in_channel=in_channel,
                                            use_highway=False)(decoded_tensor)
        decoded_tensor = HighwayResnetBlock(init_filters * (2 ** encode_i),
                                            use_highway=True)(decoded_tensor)
        decoded_tensor = HighwayResnetEncoder(init_filters * (2 ** encode_i),
                                              use_highway=True)(decoded_tensor)

    model_output = []

    if include_validity is True:
        validity = HighwayResnetBlock(out_channel=1,
                                      init_channel=init_filters *
                                      (2 ** encode_i),
                                      use_highway=False)(decoded_tensor)
        validity = activations.sigmoid(validity)
        model_output.append(validity)
    if include_classifier is True:
        predictions = layers.GlobalAveragePooling2D()(decoded_tensor)
        predictions = layers.Dense(1024)(predictions)
        predictions = base_act(predictions)
        predictions = layers.Dropout(0.5)(predictions)
        predictions = layers.Dense(1024)(predictions)
        predictions = base_act(predictions)
        predictions = layers.Dropout(0.5)(predictions)
        predictions = layers.Dense(
            num_class, activation='sigmoid')(predictions)
        model_output.append(predictions)

    return Model(model_input, model_output)
