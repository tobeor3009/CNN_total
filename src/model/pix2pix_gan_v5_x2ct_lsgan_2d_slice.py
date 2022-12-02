import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError

from .util.lsgan import to_real_loss, to_fake_loss
from .util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()
CHECK_SLICE_NUM = 8


class Pix2PixGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        ct_size,
        to_real_loss=to_real_loss,
        to_fake_loss=to_fake_loss,
        lambda_image=1,
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_image = lambda_image
        self.ct_size = ct_size
        self.slice_shape = (-1, self.ct_size, self.ct_size, 1)
        self.to_real_loss = to_real_loss
        self.to_fake_loss = to_fake_loss

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss=base_image_loss_fn,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1,
        lambda_disc=0.1
    ):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.image_loss = image_loss
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.lambda_clip = lambda_clip
        self.lambda_disc = lambda_disc
        self.disc_real_metric = tf.metrics.Accuracy()
        self.disc_fake_metric = tf.metrics.Accuracy()
        self.gen_fake_metric = tf.metrics.Accuracy()
        super(Pix2PixGan, self).compile()
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    # def build(self, input_shape):
    #     self.disc_real_metric.update_state([[1], [1]], [[1], [1]])
    #     self.disc_fake_metric.update_state([[0], [0]], [[0], [0]])
    #     self.gen_fake_metric.update_state([[1], [1]], [[1], [1]])

    def get_metric_result(self, metric):
        result = metric.result()
        metric.reset_states()
        return result

    def set_gen_metric(self, fake_data):
        self.gen_fake_metric.update_state(tf.ones_like(fake_data) >= 0.5,
                                          fake_data >= 0.5)

    def set_disc_metric(self, real_data, fake_data):
        self.disc_real_metric.update_state(tf.ones_like(real_data) >= 0.5,
                                           real_data >= 0.5)
        self.disc_fake_metric.update_state(tf.zeros_like(fake_data) < 0.5,
                                           fake_data < 0.5)

    def get_random_slice_3d(self, data_3d, slice_rand_idx, slice_num):

        slice_data_3d = data_3d[:, slice_rand_idx:slice_rand_idx + slice_num]
        slice_data_3d = backend.permute_dimensions(slice_data_3d,
                                                   (0, 4, 1, 2, 3))
        slice_data_3d = backend.reshape(slice_data_3d, self.slice_shape)
        return slice_data_3d

    def disc_3d_to_batch_size(self, disc_3d, slice_num):
        batch_disc_3d = backend.reshape(disc_3d, (-1, slice_num))
        batch_disc_3d = backend.mean(batch_disc_3d, axis=1)
        return batch_disc_3d
    # @tf.function

    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        slice_rand_idx = tf.random.uniform(shape=(), minval=0, maxval=self.ct_size - CHECK_SLICE_NUM,
                                           dtype=tf.int32)
        real_x, real_y = batch_data
        slice_real_y = self.get_random_slice_3d(real_y,
                                                slice_rand_idx, CHECK_SLICE_NUM)
        # real_x = [real_x[..., 0:1], real_x[..., 1:]]
        # disc_real_input = backend.concatenate([real_y, real_y])

        # disc_real_ratio = 0.8 - disc_real_acc
        # disc_real_ratio = tf.clip_by_value(disc_real_ratio, 0, 1)
        # disc_fake_ratio = 0.8 - disc_fake_acc
        # disc_fake_ratio = tf.clip_by_value(disc_fake_ratio, 0, 1)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:
            # another domain mapping
            fake_y = self.generator(real_x)
            slice_fake_y = self.get_random_slice_3d(fake_y,
                                                    slice_rand_idx, CHECK_SLICE_NUM)
            # discriminator loss
            disc_real_y = self.discriminator(slice_real_y, training=True)
            disc_fake_y = self.discriminator(slice_fake_y, training=True)

            disc_real_loss = self.to_real_loss(disc_real_y)
            disc_fake_loss = self.to_fake_loss(disc_fake_y)
            disc_loss = self.disc_3d_to_batch_size(disc_real_loss + disc_fake_loss,
                                                   CHECK_SLICE_NUM) / 2
        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(disc_loss,
                                        self.discriminator.trainable_variables)
        if self.apply_adaptive_gradient_clipping:
            disc_grads = adaptive_gradient_clipping(disc_grads,
                                                    self.discriminator.trainable_variables,
                                                    lambda_clip=self.lambda_clip)
        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )
        self.set_disc_metric(disc_real_y, disc_fake_y)
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            slice_fake_y = self.get_random_slice_3d(fake_y,
                                                    slice_rand_idx, CHECK_SLICE_NUM)
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            gen_disc_fake_y = self.discriminator(slice_fake_y)
            gen_adverserial_loss = self.to_real_loss(gen_disc_fake_y)
            gen_adverserial_loss = self.disc_3d_to_batch_size(gen_adverserial_loss,
                                                              CHECK_SLICE_NUM)
            total_generator_loss = gen_loss_in_real_y * self.lambda_image + \
                gen_adverserial_loss * self.lambda_disc

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(total_generator_loss,
                                      self.generator.trainable_variables)
        if self.apply_adaptive_gradient_clipping is True:
            gen_grads = adaptive_gradient_clipping(
                gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        self.set_gen_metric(gen_disc_fake_y)

        disc_real_acc = self.get_metric_result(self.disc_real_metric)
        disc_fake_acc = self.get_metric_result(self.disc_fake_metric)
        gen_fake_acc = self.get_metric_result(self.gen_fake_metric)

        return {
            "total_generator_loss": total_generator_loss,
            "generator_loss_in_real_y": gen_loss_in_real_y,
            "gen_disc_loss": gen_adverserial_loss,
            "disc_real_loss": disc_real_loss,
            "disc_fake_loss": disc_fake_loss,
            "disc_real_acc": disc_real_acc,
            "disc_fake_acc": disc_fake_acc,
            "gen_fake_acc": gen_fake_acc
        }
