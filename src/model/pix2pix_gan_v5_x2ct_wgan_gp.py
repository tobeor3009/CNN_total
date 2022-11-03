import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.losses import MeanAbsoluteError

from .util.wgan_gp import to_real_loss, to_fake_loss, gradient_penalty
from .util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class Pix2PixGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        lambda_image=1,
        lambda_disc=0.1,
        gp_weight=10,
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_image = lambda_image
        self.lambda_disc = lambda_disc
        self.gp_weight = gp_weight

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss=base_image_loss_fn,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.image_loss = image_loss
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.lambda_clip = lambda_clip
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    # @tf.function
    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, real_y = batch_data
        lower_rand_idx = tf.random.uniform(shape=(), minval=0, maxval=127,
                                           dtype=tf.int32)
        upper_rand_idx = tf.random.uniform(shape=(), minval=128, maxval=255,
                                           dtype=tf.int32)
        real_lower_slice = real_y[:, lower_rand_idx]
        real_upper_slice = real_y[:, upper_rand_idx]
        real_slice = tf.concat([real_lower_slice, real_upper_slice],
                               axis=-1)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator(real_x)
            fake_lower_slice = fake_y[:, lower_rand_idx]
            fake_upper_slice = fake_y[:, upper_rand_idx]
            fake_slice = tf.concat([fake_lower_slice, fake_upper_slice],
                                   axis=-1)
            # discriminator loss
            disc_real_y = self.discriminator(real_slice, training=True)
            disc_fake_y = self.discriminator(fake_slice, training=True)

            disc_real_y_loss = to_real_loss(disc_real_y)
            disc_fake_y_loss = to_fake_loss(disc_fake_y)
            gp = gradient_penalty(self.discriminator, real_slice, fake_slice,
                                  gp_weight=self.gp_weight, mode="2d")
            disc_y_loss = disc_real_y_loss + disc_fake_y_loss + gp
        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(disc_y_loss,
                                        self.discriminator.trainable_variables)
        if self.apply_adaptive_gradient_clipping:
            disc_grads = adaptive_gradient_clipping(disc_grads,
                                                    self.discriminator.trainable_variables,
                                                    lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            fake_lower_slice = fake_y[:, lower_rand_idx]
            fake_upper_slice = fake_y[:, upper_rand_idx]
            fake_slice = tf.concat([fake_lower_slice, fake_upper_slice],
                                   axis=-1)
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            disc_fake_y = self.discriminator(fake_slice)
            gen_adverserial_loss = to_real_loss(disc_fake_y)

            total_generator_loss = gen_loss_in_real_y * self.lambda_image + \
                gen_adverserial_loss * self.lambda_disc

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(
            total_generator_loss, self.generator.trainable_variables)
        if self.apply_adaptive_gradient_clipping is True:
            gen_grads = adaptive_gradient_clipping(
                gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_generator_loss": total_generator_loss,
            "generator_loss_in_real_y": gen_loss_in_real_y,
            "gen_disc_loss": gen_adverserial_loss,
            "disc_loss": disc_y_loss,
            "disc_real_loss": disc_real_y_loss,
            "disc_fake_loss": disc_fake_y_loss
        }
