import tensorflow as tf
from tensorflow.keras import backend as keras_backend


def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)


def to_real_wasserstein_loss(img_tensor):
    real_labels = tf.ones_like(img_tensor)
    return wasserstein_loss(real_labels, img_tensor)


def to_fake_wasserstein_loss(img_tensor):
    fake_labels = -tf.ones_like(img_tensor)
    return wasserstein_loss(fake_labels, img_tensor)


@tf.function
def gradient_penalty(discriminator, batch_size,
                     real_images, fake_images, gp_weight,
                     mode="2d", smooth=1e-7):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    if mode == "2d":
        alpha_shape = [batch_size, 1, 1, 1]
    elif mode == "3d":
        alpha_shape = [batch_size, 1, 1, 1, 1]
    # Get the interpolated image
    alpha = tf.random.normal(alpha_shape, 0.0, 1.0)
    interpolated_real = real_images
    interpolated_fake = real_images * alpha + fake_images * (1 - alpha)
    interpolated = keras_backend.concatenate(
        [interpolated_real, interpolated_fake], axis=-1)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)
        loss_fake_grad = to_fake_wasserstein_loss(pred)
    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(loss_fake_grad, [interpolated])
    # 3. Calculate the norm of the gradients.
    gp = grads * gp_weight
    return gp

# @tf.function
# def gradient_penalty(discriminator, batch_size,
#                      real_images, fake_images,
#                      mode="2d", smooth=1e-7):
#     """ Calculates the gradient penalty.

#     This loss is calculated on an interpolated image
#     and added to the discriminator loss.
#     """
#     if mode == "2d":
#         alpha_shape = [batch_size, 1, 1, 1]
#     elif mode == "3d":
#         alpha_shape = [batch_size, 1, 1, 1, 1]
#     # Get the interpolated image
#     alpha = tf.random.normal(alpha_shape, 0.0, 1.0)
#     diff = fake_images - real_images
#     interpolated = real_images + alpha * diff

#     with tf.GradientTape() as gp_tape:
#         gp_tape.watch(interpolated)
#         # 1. Get the discriminator output for this interpolated image.
#         pred = discriminator(interpolated, training=True)

#     # 2. Calculate the gradients w.r.t to this interpolated image.
#     grads = gp_tape.gradient(pred, [interpolated])[0]
#     # 3. Calculate the norm of the gradients.
#     norm = tf.sqrt(smooth + tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
#     gp = tf.reduce_mean((norm - 1.0) ** 2)
#     return gp
