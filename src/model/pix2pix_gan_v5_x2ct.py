import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.losses import MeanAbsoluteError

from .util.wgan_gp import base_generator_loss_deceive_discriminator, \
    base_discriminator_loss_arrest_generator, gradient_penalty
from .util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class Pix2PixGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        batch_size,
        lambda_gp=1,
        lambda_disc=0.1
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp
        self.lambda_disc = lambda_disc

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss=base_image_loss_fn,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
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
        real_x = [real_x[..., 0:1], real_x[..., 1:]]
        disc_real_input = backend.concatenate([real_y, real_y])
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator(real_x)
            # discriminator loss
            disc_fake_input = backend.concatenate([real_y, fake_y])
            disc_real_y = self.discriminator(disc_real_input, training=True)
            disc_fake_y = self.discriminator(disc_fake_input, training=True)
            disc_loss = self.discriminator_loss_arrest_generator(disc_real_y,
                                                                 disc_fake_y)
            # gp = gradient_penalty(self.discriminator,
            #                       self.batch_size, real_y, fake_y,
            #                       mode="2d")
            # disc_loss += gp * self.lambda_gp
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
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            disc_fake_input = backend.concatenate([real_y, fake_y])
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            disc_fake_y = self.discriminator(disc_fake_input)
            gen_adverserial_loss = self.generator_loss_deceive_discriminator(
                disc_fake_y)

            total_generator_loss = gen_loss_in_real_y + \
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
            "disc_loss": disc_loss
        }
