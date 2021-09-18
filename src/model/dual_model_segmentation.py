import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal
import numpy as np

from .util.lsgan import base_generator_loss_deceive_discriminator, base_discriminator_loss_arrest_generator
from .util.grad_clip import adaptive_gradient_clipping
from .util.gan_loss import rgb_color_histogram_loss

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)


class DualModelSegmentation(Model):
    def __init__(self,
                 generator_A=None,
                 generator_B=None):
        super(DualModelSegmentation, self).__init__()

        self.generator_A = generator_A
        self.generator_B = generator_B

    def compile(self,
                generator_A_optimizer,
                generator_B_optimizer,
                image_loss,
                metrics,
                lambda_clip=0.1):
        super(DualModelSegmentation, self).compile()
        self.generator_A_optimizer = generator_A_optimizer
        self.generator_B_optimizer = generator_B_optimizer
        self.image_loss = image_loss
        self.metric = metrics[0]
        self.lambda_clip = lambda_clip

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    @tf.function
    def train_step(self, batch):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_images, mask_images = batch

        gen_A_mask = mask_images[:, :, :, :1]
        gen_B_mask = mask_images[:, :, :, 1:]
        with tf.GradientTape(persistent=True) as gen_tape:
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Decode them to fake images
            gen_A_fake_mask = self.generator_A(real_images, training=True)
            gen_B_fake_mask = self.generator_B(real_images, training=True)

            gen_A_loss = self.image_loss(gen_A_mask, gen_A_fake_mask)
            gen_A_metric = self.metric(gen_A_mask, gen_A_fake_mask)
            gen_B_loss = self.image_loss(gen_B_mask, gen_B_fake_mask)
            gen_B_metric = self.metric(gen_B_mask, gen_B_fake_mask)
        # Get the gradients for the discriminators
        gen_A_grads = gen_tape.gradient(
            gen_A_loss, self.generator_A.trainable_variables)
        gen_B_grads = gen_tape.gradient(
            gen_B_loss, self.generator_B.trainable_variables)

        cliped_gen_A_grads = adaptive_gradient_clipping(
            gen_A_grads, self.generator_A.trainable_variables, lambda_clip=self.lambda_clip)
        cliped_gen_B_grads = adaptive_gradient_clipping(
            gen_B_grads, self.generator_B.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.generator_A_optimizer.apply_gradients(
            zip(cliped_gen_A_grads, self.generator_A.trainable_variables)
        )
        # Update the weights of the discriminators
        self.generator_B_optimizer.apply_gradients(
            zip(cliped_gen_B_grads, self.generator_B.trainable_variables)
        )
        return {
            "gen_A_loss": gen_A_loss,
            "gen_B_loss": gen_B_loss,
            "gen_A_metric": gen_A_metric,
            "gen_B_metric": gen_B_metric,
        }

    @tf.function
    def test_step(self, batch):
        real_images, mask_images = batch
        gen_A_mask = mask_images[:, :, :, :1]
        gen_B_mask = mask_images[:, :, :, 1:]

        gen_A_fake_mask = self.generator_A(real_images, training=False)
        gen_B_fake_mask = self.generator_B(real_images, training=False)

        gen_A_loss = self.image_loss(gen_A_mask, gen_A_fake_mask)
        gen_A_metric = self.metric(gen_A_mask, gen_A_fake_mask)
        gen_B_loss = self.image_loss(gen_B_mask, gen_B_fake_mask)
        gen_B_metric = self.metric(gen_B_mask, gen_B_fake_mask)
        return {
            "gen_A_loss": gen_A_loss,
            "gen_B_loss": gen_B_loss,
            "gen_A_metric": gen_A_metric,
            "gen_B_metric": gen_B_metric,
        }
