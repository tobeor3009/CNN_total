import tensorflow as tf
from tensorflow.keras import backend as keras_backend


def compute_l2_norm(tensor):

    squared = keras_backend.square(tensor)
    l2_norm = keras_backend.sum(squared)
    l2_norm = keras_backend.sqrt(l2_norm)

    return l2_norm


@tf.function
def active_gradient_clipping(grad_list, trainable_variable_list, lambda_clip=10):

    cliped_grad_list = []

    for grad, trainable_variable in zip(grad_list, trainable_variable_list):
        grad_l2_norm = compute_l2_norm(grad)
        trainable_variable_l2_norm = compute_l2_norm(trainable_variable)

        clip_value = lambda_clip * \
            (trainable_variable_l2_norm / grad_l2_norm)
        cliped_grad = keras_backend.clip(grad, -clip_value, clip_value)

        cliped_grad_list.append(cliped_grad)

    return cliped_grad_list
