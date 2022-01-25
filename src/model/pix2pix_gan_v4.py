import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError

from .util.gan_loss import rgb_color_histogram_loss
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
        gp_weight=10.0,
        lambda_histogram=0.1
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = gp_weight
        self.lambda_histogram = lambda_histogram

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        batch_size,
        image_loss=base_image_loss_fn,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
        adaptive_gradient_clipping=True,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.image_loss = image_loss
        self.batch_size = batch_size
        self.adaptive_gradient_clipping = adaptive_gradient_clipping
        self.lambda_clip = lambda_clip
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    @tf.function
    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, real_y = batch_data
        real_y_slice = real_y[..., 127, :]
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator(real_x)
            fake_y_slice = fake_y[..., 127, :]
            # Discriminator output
            disc_real_y = self.discriminator(real_y_slice, training=True)
            disc_fake_y = self.discriminator(fake_y_slice, training=True)

            # Discriminator loss
            disc_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)
            disc_fake_y_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_y, fake_y)
            disc_loss += self.gp_weight * disc_fake_y_gradient_panalty

        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        cliped_disc_grads = adaptive_gradient_clipping(
            disc_grads, self.discriminator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(cliped_disc_grads, self.discriminator.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            fake_y_slice = fake_y[..., 127, :]

            # Discriminator output
            disc_fake_y = self.discriminator(fake_y_slice)
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            gen_loss_adverserial_loss = self.generator_loss_deceive_discriminator(
                disc_fake_y)
            total_generator_loss = gen_loss_in_real_y + \
                gen_loss_adverserial_loss

            if self.lambda_histogram > 0:
                gen_histo_loss_in_real_y = self.lambda_histogram * \
                    rgb_color_histogram_loss(real_y, fake_y)
                total_generator_loss += gen_histo_loss_in_real_y
            else:
                gen_histo_loss_in_real_y = 0
        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(
            total_generator_loss, self.generator.trainable_variables)
        if self.adaptive_gradient_clipping is True:
            cliped_gen_grads = adaptive_gradient_clipping(
                gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(cliped_gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_generator_loss": total_generator_loss,
            "generator_loss_in_real_y": gen_loss_in_real_y,
            "generator_adverserial_loss": gen_loss_adverserial_loss,
            "generator_histo_loss": gen_histo_loss_in_real_y,
            "discriminator_loss": disc_loss,
        }
