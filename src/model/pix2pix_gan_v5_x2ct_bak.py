import numpy as np
import random
import tensorflow as tf
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
        discriminator_2d,
        discriminator_3d,
        batch_size,
        lambda_disc=0.1
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator_2d = discriminator_2d
        self.discriminator_3d = discriminator_3d
        self.batch_size = batch_size
        self.lambda_disc = lambda_disc

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss=base_image_loss_fn,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
        apply_adaptive_gradient_clipping=True,
        gp_weight=10.0,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.image_loss = image_loss
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.gp_weight = gp_weight
        self.lambda_clip = lambda_clip
        self.ct_slice_num = self.generator.output_shape[-1]
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    # @tf.function
    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, real_y = batch_data
        slice_index_list = random.sample(range(self.ct_slice_num), 3)
        y_1 = real_y[..., slice_index_list[0]]
        y_2 = real_y[..., slice_index_list[1]]
        y_3 = real_y[..., slice_index_list[2]]
        real_y_2d = keras_backend.concatenate([y_1, y_2, y_3], axis=0)
        real_y_2d = keras_backend.expand_dims(real_y_2d, axis=-1)
        real_x = [real_x[..., 0:1], real_x[..., 1:]]
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator(real_x)
            # Discriminator output
            fake_y_1 = fake_y[..., slice_index_list[0]]
            fake_y_2 = fake_y[..., slice_index_list[1]]
            fake_y_3 = fake_y[..., slice_index_list[2]]
            fake_y_2d = keras_backend.concatenate(
                [fake_y_1, fake_y_2, fake_y_3], axis=0)
            fake_y_2d = keras_backend.expand_dims(fake_y_2d, axis=-1)

            # discriminator_2d loss
            disc_real_y_2d = self.discriminator_2d(real_y_2d, training=True)
            disc_fake_y_2d = self.discriminator_2d(fake_y_2d, training=True)
            disc_loss_2d = self.discriminator_loss_arrest_generator(
                disc_real_y_2d, disc_fake_y_2d)
            disc_2d_gradient_panalty = gradient_penalty(self.discriminator_2d,
                                                        self.batch_size * 3,
                                                        real_y_2d, fake_y_2d, mode="2d")
            disc_loss_2d += self.gp_weight * disc_2d_gradient_panalty
            # discriminator_3d loss
            disc_real_y_3d = self.discriminator_3d(real_y, training=True)
            disc_fake_y_3d = self.discriminator_3d(fake_y, training=True)
            disc_loss_3d = self.discriminator_loss_arrest_generator(
                disc_real_y_3d, disc_fake_y_3d)
            disc_3d_gradient_panalty = gradient_penalty(self.discriminator_3d,
                                                        self.batch_size,
                                                        real_y, fake_y, mode="2d")
            disc_loss_3d += self.gp_weight * disc_3d_gradient_panalty
        # Get the gradients for the discriminators
        disc_grads_2d = disc_tape.gradient(
            disc_loss_2d, self.discriminator_2d.trainable_variables)
        disc_grads_3d = disc_tape.gradient(
            disc_loss_3d, self.discriminator_3d.trainable_variables)
        if self.apply_adaptive_gradient_clipping:
            disc_grads_2d = adaptive_gradient_clipping(
                disc_grads_2d, self.discriminator_2d.trainable_variables, lambda_clip=self.lambda_clip)
            disc_grads_3d = adaptive_gradient_clipping(
                disc_grads_3d, self.discriminator_3d.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads_2d, self.discriminator_2d.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads_3d, self.discriminator_3d.trainable_variables)
        )
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            fake_y_1 = fake_y[..., slice_index_list[0]]
            fake_y_2 = fake_y[..., slice_index_list[1]]
            fake_y_3 = fake_y[..., slice_index_list[2]]
            fake_y_2d = keras_backend.concatenate(
                [fake_y_1, fake_y_2, fake_y_3], axis=0)
            fake_y_2d = keras_backend.expand_dims(fake_y_2d, axis=-1)
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss 2d
            disc_fake_y_2d = self.discriminator_2d(fake_y_2d)
            gen_adverserial_loss_2d = self.generator_loss_deceive_discriminator(
                disc_fake_y_2d)
            # Generator adverserial loss 3d
            disc_fake_y_3d = self.discriminator_3d(fake_y)
            gen_adverserial_loss_3d = self.generator_loss_deceive_discriminator(
                disc_fake_y_3d)

            total_generator_loss = gen_loss_in_real_y + \
                gen_adverserial_loss_2d * self.lambda_disc + \
                gen_adverserial_loss_3d * self.lambda_disc

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
            "gen_disc_loss_2d": gen_adverserial_loss_2d,
            "gen_disc_loss_3d": gen_adverserial_loss_3d,
            "disc_loss_2d": disc_loss_2d,
            "disc_loss_3d": disc_loss_3d
        }
