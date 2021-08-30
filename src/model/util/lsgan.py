import tensorflow as tf

mse = tf.losses.MeanSquaredError()


def base_generator_loss_deceive_discriminator(fake_img):
    f_loss = mse(tf.ones_like(fake_img), fake_img)
    return f_loss


def base_discriminator_loss_arrest_generator(real_img, fake_img):
    r_loss = mse(tf.ones_like(real_img), real_img)
    f_loss = mse(tf.zeros_like(fake_img), fake_img)
    loss = r_loss + f_loss
    return loss
