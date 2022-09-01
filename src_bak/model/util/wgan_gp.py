import tensorflow as tf
from tensorflow.keras import backend as keras_backend


def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)


def to_real_loss(img_tensor):
    return -tf.reduce_mean(img_tensor)


def to_fake_loss(img_tensor):
    return tf.reduce_mean(img_tensor)


def interpolate(label_shape, real, fake=None, mode="2d"):
    if mode == "2d":
        alpha_shape = [label_shape[0], 1, 1, 1]
    elif mode == "3d":
        alpha_shape = [label_shape[0], 1, 1, 1, 1]
    beta = keras_backend.random_normal(shape=label_shape)
    if fake is None:
        fake = real + 0.5 * tf.sqrt(tf.math.reduce_variance(real)) * beta
    alpha = tf.random.normal(alpha_shape, 0.0, 1.0)
    inter = real + alpha * (fake - real)
    return inter


@tf.function
def gradient_penalty(disc, real, fake, gp_weight,
                     mode="2d", smooth=1e-7):
    label_shape = tf.shape(real)
    inter = interpolate(label_shape, real, fake, mode=mode)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inter)
        # 1. Get the discriminator output for this interpolated image.
        # disc.output = [validity, class]
        pred = disc(inter, training=True)
        if isinstance(pred, list):
            pred = pred[0]
    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, inter,
                             output_gradients=tf.ones_like(pred))
    grads = tf.reshape(grads, [label_shape[0], -1])
    norm = tf.norm(grads + smooth, axis=-1)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    # 3. Calculate the norm of the gradients.
    gp = gp * gp_weight
    return gp


# @tf.function
# def gradient_penalty(discriminator, real_images, fake_images, gp_weight,
#                      mode="2d", smooth=1e-7):
#     """ Calculates the gradient penalty.

#     This loss is calculated on an interpolated image
#     and added to the discriminator loss.
#     """
#     label_shape = tf.shape(real_images)
#     batch_size = label_shape[0]
#     if mode == "2d":
#         alpha_shape = [batch_size, 1, 1, 1]
#     elif mode == "3d":
#         alpha_shape = [batch_size, 1, 1, 1, 1]
#     # Get the interpolated image
#     alpha = tf.random.normal(alpha_shape, 0.0, 1.0)
#     interpolated = real_images + (fake_images - real_images) * alpha
#     with tf.GradientTape() as gp_tape:
#         gp_tape.watch(interpolated)
#         # 1. Get the discriminator output for this interpolated image.
#         pred = discriminator(interpolated, training=True)[0]
#     # 2. Calculate the gradients w.r.t to this interpolated image.
#     grads = gp_tape.gradient(pred, [interpolated],
#                              output_gradients=tf.ones_like(pred))
#     grads = tf.reshape(grads, [batch_size, -1])
#     norm = tf.sqrt(smooth + tf.reduce_sum(tf.square(grads), axis=1))
#     gp = tf.reduce_mean((norm - 1.0) ** 2)
#     # 3. Calculate the norm of the gradients.
#     gp = gp * gp_weight
#     return gp

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
