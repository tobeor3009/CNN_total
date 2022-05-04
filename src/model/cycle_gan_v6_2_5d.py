import tensorflow as tf
from tensorflow.keras import backend as keras_backend
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal

from .util.lsgan import base_generator_loss_deceive_discriminator, base_discriminator_loss_arrest_generator
from .util.grad_clip import adaptive_gradient_clipping

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class CycleGan(Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_gen_2_disc=1.0,
        lambda_cycle=10.0,
        lambda_identity=0.5,
        gp_weight=10.0
    ):
        super(CycleGan, self).__init__()

        self.generator_G = generator_G
        self.generator_F = generator_F
        self.discriminator_X = discriminator_X
        self.discriminator_Y = discriminator_Y

        self.lambda_gen_2_disc = lambda_gen_2_disc
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.turn_on_identity_loss = True
        self.turn_on_discriminator_on_identity = False
        self.gp_weight = gp_weight

    def compile(
        self,
        generator_G_optimizer,
        generator_F_optimizer,
        discriminator_X_optimizer,
        discriminator_Y_optimizer,
        image_loss_fn=base_image_loss_fn,
        generator_loss_deceive_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
        identity_loss=True,
        active_gradient_clip=True,
        lambda_clip=0.1,
    ):
        super(CycleGan, self).compile()
        self.generator_G_optimizer = generator_G_optimizer
        self.generator_F_optimizer = generator_F_optimizer
        self.discriminator_X_optimizer = discriminator_X_optimizer
        self.discriminator_Y_optimizer = discriminator_Y_optimizer
        self.generator_loss_deceive_discriminator = generator_loss_deceive_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.cycle_loss_fn = image_loss_fn
        self.identity_loss_fn = image_loss_fn
        self.identity_loss = identity_loss
        self.active_gradient_clip = active_gradient_clip
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
        _, H, W, _ = keras_backend.int_shape(real_x)
        upper_real_x, middle_real_x, below_real_x = real_x[...,
                                                           0:3], real_x[..., 1:4], real_x[..., 2:5]
        upper_real_y, middle_real_y, below_real_y = real_y[...,
                                                           0:3], real_y[..., 1:4], real_y[..., 2:5]
        disc_middle_real_x = keras_backend.reshape(
            middle_real_x, (-1, H, W, 1))
        disc_middle_real_y = keras_backend.reshape(
            middle_real_y, (-1, H, W, 1))

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as disc_tape:

            # another domain mapping
            upper_fake_x = self.generator_F(upper_real_y)
            middle_fake_x = self.generator_F(middle_real_y)
            below_fake_x = self.generator_F(below_real_y)
            fake_x = keras_backend.concatenate([upper_fake_x,
                                                middle_fake_x,
                                                below_fake_x], axis=-1)
            fake_x = keras_backend.reshape(fake_x, (-1, H, W, 1))

            upper_fake_y = self.generator_G(upper_real_x)
            middle_fake_y = self.generator_G(middle_real_x)
            below_fake_y = self.generator_G(below_real_x)
            fake_y = keras_backend.concatenate([upper_fake_y,
                                                middle_fake_y,
                                                below_fake_y], axis=-1)
            fake_y = keras_backend.reshape(fake_y, (-1, H, W, 1))
            # back to original domain mapping
            cycle_x = self.generator_F(fake_y)
            cycle_y = self.generator_G(fake_x)

            # Discriminator output
            disc_real_x = self.discriminator_X(
                disc_middle_real_x, training=True)
            disc_fake_x = self.discriminator_X(fake_x,
                                               training=True)
            disc_cycle_x = self.discriminator_X(cycle_x, training=True)

            disc_real_y = self.discriminator_Y(
                disc_middle_real_y, training=True)
            disc_fake_y = self.discriminator_Y(fake_y,
                                               training=True)
            disc_cycle_y = self.discriminator_Y(cycle_y, training=True)

            if self.identity_loss is True:
                # Identity mapping
                same_x = self.generator_F(middle_real_x)
                same_y = self.generator_G(middle_real_y)

                disc_same_x = self.discriminator_X(same_x, training=True)
                disc_same_y = self.discriminator_Y(same_y, training=True)

                disc_X_identity_loss = self.discriminator_loss_arrest_generator(
                    disc_real_x, disc_same_x)
                disc_Y_identity_loss = self.discriminator_loss_arrest_generator(
                    disc_real_y, disc_same_y)

            else:
                disc_X_identity_loss = 0
                disc_Y_identity_loss = 0

            # Discriminator loss
            disc_X_fake_loss = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            disc_Y_fake_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)
            disc_X_cycle_loss = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_cycle_x)
            disc_Y_cycle_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_cycle_y)

            disc_X_total_loss = disc_X_fake_loss + disc_X_cycle_loss + \
                disc_X_identity_loss * self.lambda_identity
            disc_Y_total_loss = disc_Y_fake_loss + disc_Y_cycle_loss + \
                disc_Y_identity_loss * self.lambda_identity

        # Get the gradients for the discriminators
        disc_X_grads = disc_tape.gradient(
            disc_X_total_loss, self.discriminator_X.trainable_variables)
        disc_Y_grads = disc_tape.gradient(
            disc_Y_total_loss, self.discriminator_Y.trainable_variables)

        if self.active_gradient_clip is True:
            # Apply Active Gradient Clipping on discriminator's grad
            disc_X_grads = adaptive_gradient_clipping(
                disc_X_grads, self.discriminator_X.trainable_variables, lambda_clip=self.lambda_clip)
            disc_Y_grads = adaptive_gradient_clipping(
                disc_Y_grads, self.discriminator_Y.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.discriminator_X.trainable_variables)
        )
        self.discriminator_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.discriminator_Y.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as gen_tape:

            # another domain mapping
            upper_fake_x = self.generator_F(upper_real_y, training=True)
            middle_fake_x = self.generator_F(middle_real_y, training=True)
            below_fake_x = self.generator_F(below_real_y, training=True)
            fake_x = keras_backend.concatenate([upper_fake_x,
                                                middle_fake_x,
                                                below_fake_x], axis=-1)
            fake_x = keras_backend.reshape(fake_x, (-1, H, W, 1))
            upper_fake_y = self.generator_F(upper_real_x, training=True)
            middle_fake_y = self.generator_F(middle_real_x, training=True)
            below_fake_y = self.generator_F(below_real_x, training=True)
            fake_y = keras_backend.concatenate([upper_fake_y,
                                                middle_fake_y,
                                                below_fake_y], axis=-1)
            fake_y = keras_backend.reshape(fake_y, (-1, H, W, 1))
            # back to original domain mapping
            cycle_x = self.generator_F(fake_y, training=True)
            cycle_y = self.generator_G(fake_x, training=True)

            # Discriminator output
            disc_fake_x = self.discriminator_X(fake_x)
            disc_fake_y = self.discriminator_Y(fake_y)

            disc_cycle_x = self.discriminator_X(cycle_x)
            disc_cycle_y = self.discriminator_Y(cycle_y)

            if self.identity_loss is True:
                # Identity mapping
                same_x = self.generator_F(middle_real_x, training=True)
                same_y = self.generator_G(middle_real_y, training=True)

                disc_same_x = self.discriminator_X(same_x)
                disc_same_y = self.discriminator_Y(same_y)

                gen_G_identity_image_loss = (
                    self.identity_loss_fn(real_x[..., 2:3], same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
                gen_F_identity_image_loss = (
                    self.identity_loss_fn(real_y[..., 2:3], same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
                )

                # Generator adverserial loss
                gen_G_identity_disc_loss = self.generator_loss_deceive_discriminator(
                    disc_same_y)
                gen_F_identity_disc_loss = self.generator_loss_deceive_discriminator(
                    disc_same_x)
            else:
                gen_G_identity_image_loss = 0
                gen_F_identity_image_loss = 0
                gen_G_identity_disc_loss = 0
                gen_F_identity_disc_loss = 0

            # Generator cycle loss
            gen_G_cycle_image_loss = self.cycle_loss_fn(
                real_y, cycle_y) * self.lambda_cycle
            gen_F_cycle_image_loss = self.cycle_loss_fn(
                real_x, cycle_x) * self.lambda_cycle

            gen_G_fake_disc_loss = self.generator_loss_deceive_discriminator(
                disc_fake_y)
            gen_F_fake_disc_loss = self.generator_loss_deceive_discriminator(
                disc_fake_x)

            gen_G_cycle_disc_loss = self.generator_loss_deceive_discriminator(
                disc_cycle_y)
            gen_F_cycle_disc_loss = self.generator_loss_deceive_discriminator(
                disc_cycle_x)

            gen_G_image_loss = gen_G_cycle_image_loss + gen_G_identity_image_loss
            gen_F_image_loss = gen_F_cycle_image_loss + gen_F_identity_image_loss

            gen_G_disc_loss = (gen_G_fake_disc_loss + gen_G_cycle_disc_loss +
                               gen_G_identity_disc_loss) * self.lambda_gen_2_disc
            gen_F_disc_loss = (gen_F_fake_disc_loss + gen_F_cycle_disc_loss +
                               gen_F_identity_disc_loss) * self.lambda_gen_2_disc

            # Total generator loss
            gen_G_total_loss = gen_G_image_loss + gen_G_disc_loss
            gen_F_total_loss = gen_F_image_loss + gen_F_disc_loss

        # Get the gradients for the generators
        gen_G_grads = gen_tape.gradient(
            gen_G_total_loss, self.generator_G.trainable_variables)
        gen_F_grads = gen_tape.gradient(
            gen_F_total_loss, self.generator_F.trainable_variables)

        if self.active_gradient_clip is True:
            # Apply Active Gradient Clipping on generator's grad
            gen_G_grads = adaptive_gradient_clipping(
                gen_G_grads, self.generator_G.trainable_variables, lambda_clip=self.lambda_clip)
            gen_F_grads = adaptive_gradient_clipping(
                gen_F_grads, self.generator_F.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_G_optimizer.apply_gradients(
            zip(gen_G_grads, self.generator_G.trainable_variables)
        )
        self.generator_F_optimizer.apply_gradients(
            zip(gen_F_grads, self.generator_F.trainable_variables)
        )
        return {
            "total_loss_G": gen_G_total_loss,
            "total_loss_F": gen_F_total_loss,
            "D_X_loss": disc_X_total_loss,
            "D_Y_loss": disc_Y_total_loss,
            "generator_G_loss": gen_G_fake_disc_loss,
            "generator_F_loss": gen_F_fake_disc_loss,
            "identity_loss_G": gen_G_identity_image_loss,
            "identity_loss_F": gen_F_identity_image_loss,
            "cycle_loss_G": gen_G_cycle_image_loss,
            "cycle_loss_F": gen_F_cycle_image_loss
        }
