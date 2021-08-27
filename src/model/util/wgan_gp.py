import tensorflow as tf
from tensorflow.keras import backend as keras_backend

# Define the loss function for the generators


def base_generator_loss_deceive_discriminator(fake_img):
    return -tf.reduce_mean(fake_img)

# Define the loss function for the discriminators


def base_discriminator_loss_arrest_generator(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def compute_l2_norm(tensor):

    squared = keras_backend.square(tensor)
    l2_norm = keras_backend.sum(squared)
    l2_norm = keras_backend.sqrt(l2_norm)

    return l2_norm


@tf.function
def gradient_penalty(discriminator, batch_size, real_images, fake_images):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


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
