import tensorflow as tf


# def to_real_loss(label):
#     mse = (tf.ones_like(label) - label) ** 2
#     mse = tf.reduce_mean(mse)
#     return mse


# def to_fake_loss(label):
#     mse = (tf.zeros_like(label) - label) ** 2
#     mse = tf.reduce_mean(mse)
#     return mse


def to_real_loss(label):
    mse = tf.reduce_mean((1 - label) ** 2)
    return mse


def to_fake_loss(label):
    mse = tf.reduce_mean(label ** 2)
    return mse