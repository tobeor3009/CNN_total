import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.losses import CategoricalCrossentropy

from .util.wgan_gp import base_generator_loss_deceive_discriminator, \
    base_discriminator_loss_arrest_generator, gradient_penalty
from .util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = CategoricalCrossentropy(label_smoothing=0.01)


class StarGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        label_mode="concatenate",
        lambda_reconstruct=10.0,
        lambda_identity=0.5,
        gp_weight=10.0
    ):
        super(StarGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.label_mode = label_mode
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_identity = lambda_identity
        self.turn_on_identity_loss = True
        self.gp_weight = gp_weight

    def compile(
        self,
        batch_size,
        image_shape,
        label_num,
        generator_optimizer,
        discriminator_optimizer,
        image_loss_fn=base_image_loss_fn,
        class_loss_fn=base_class_loss_fn,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
        active_gradient_clip=True,
        lambda_clip=0.1
    ):
        super(StarGan, self).compile()
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.label_num = label_num
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.reconstruct_loss_fn = image_loss_fn
        self.identity_loss_fn = image_loss_fn
        self.class_loss_fn = class_loss_fn
        self.active_gradient_clip = active_gradient_clip
        self.lambda_clip = lambda_clip
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    @tf.function
    def train_step(self, batch_data):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        image_tensor, label_tensor = batch_data
        real_x = image_tensor[0]
        real_y = image_tensor[1]
        label_x = label_tensor[0]
        label_y = label_tensor[1]

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=False) as disc_tape:

            # another domain mapping
            fake_x = self.generator([real_y, label_y, label_x])
            fake_y = self.generator([real_x, label_x, label_y])

            # back to original domain mapping
            reconstruct_x = self.generator([fake_y, label_y, label_x])
            reconstruct_y = self.generator([fake_x, label_x, label_y])

            same_x = self.generator([real_x, label_x, label_x])
            same_y = self.generator([real_y, label_y, label_y])

            # Discriminator output
            disc_real_x, label_predicted_real_x = self.discriminator(
                real_x, training=True)
            disc_fake_x, label_predicted_fake_x = self.discriminator(
                fake_x, training=True)
            disc_reconstruct_x, label_predicted_reconstruct_x = self.discriminator(
                reconstruct_x, training=True)
            disc_same_x, label_predicted_same_x = self.discriminator(
                same_x, training=True)

            disc_real_y, label_predicted_real_y = self.discriminator(
                real_y, training=True)
            disc_fake_y, label_predicted_fake_y = self.discriminator(
                fake_y, training=True)
            disc_reconstruct_y, label_predicted_reconstruct_y = self.discriminator(
                reconstruct_y, training=True)
            disc_same_y, label_predicted_same_y = self.discriminator(
                same_y, training=True)

            # Compute Discriminator class_loss_x
            disc_class_loss_real_x = self.class_loss_fn(
                label_x, label_predicted_real_x)
            disc_fake_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_fake_x)
            disc_reconstruct_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_reconstruct_x)
            disc_same_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_same_x)

            # Compute Discriminator wgan_loss_x
            disc_fake_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            disc_fake_x_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_x, fake_x)
            disc_fake_loss_x += self.gp_weight * disc_fake_x_gradient_panalty

            disc_reconstruct_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_reconstruct_x)
            disc_reconstruct_x_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_x, reconstruct_x)
            disc_reconstruct_loss_x += self.gp_weight * disc_reconstruct_x_gradient_panalty

            disc_same_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_same_x)
            disc_same_x_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_x, same_x)
            disc_same_loss_x += self.gp_weight * disc_same_x_gradient_panalty

            # Compute Discriminator class_loss_y
            disc_class_loss_real_y = self.class_loss_fn(
                label_y, label_predicted_real_y)
            disc_fake_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_fake_y)
            disc_reconstruct_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_reconstruct_y)
            disc_same_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_same_y)

            # Compute Discriminator wgan_loss_y
            disc_fake_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)
            disc_fake_y_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_y, fake_y)
            disc_fake_loss_y += self.gp_weight * disc_fake_y_gradient_panalty

            disc_reconstruct_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_reconstruct_y)
            disc_reconstruct_y_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_y, reconstruct_y)
            disc_reconstruct_loss_y += self.gp_weight * disc_reconstruct_y_gradient_panalty

            disc_same_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_same_y)
            disc_same_y_gradient_panalty = gradient_penalty(
                self.discriminator, self.batch_size, real_y, same_y)
            disc_same_loss_y += self.gp_weight * disc_same_y_gradient_panalty

            disc_total_loss_x = disc_fake_loss_x + disc_reconstruct_loss_x + \
                disc_same_loss_x + \
                disc_class_loss_real_x + disc_fake_class_loss_x +\
                disc_reconstruct_class_loss_x + disc_same_class_loss_x
            disc_total_loss_y = disc_fake_loss_y + disc_reconstruct_loss_y + \
                disc_same_loss_y + \
                disc_class_loss_real_y + disc_fake_class_loss_y + \
                disc_reconstruct_class_loss_y + disc_same_class_loss_y

            disc_total_loss = disc_total_loss_x + disc_total_loss_y

        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(
            disc_total_loss, self.discriminator.trainable_variables)
        if self.active_gradient_clip is True:
            disc_grads = adaptive_gradient_clipping(
                disc_grads, self.discriminator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:

            # another domain mapping
            fake_x = self.generator([real_y, label_y, label_x], training=True)
            fake_y = self.generator([real_x, label_x, label_y], training=True)

            # back to original domain mapping
            reconstruct_x = self.generator(
                [fake_y, label_y, label_x], training=True)
            reconstruct_y = self.generator(
                [fake_x, label_x, label_y], training=True)

            same_x = self.generator([real_x, label_x, label_x], training=True)
            same_y = self.generator([real_y, label_y, label_y], training=True)

            # Discriminator output
            disc_real_x, label_predicted_real_x = self.discriminator(real_x)
            disc_fake_x, label_predicted_fake_x = self.discriminator(fake_x)
            disc_reconstruct_x, label_predicted_reconstruct_x = self.discriminator(
                reconstruct_x)
            disc_same_x, label_predicted_same_x = self.discriminator(same_x)

            disc_real_y, label_predicted_real_y = self.discriminator(real_y)
            disc_fake_y, label_predicted_fake_y = self.discriminator(fake_y)
            disc_reconstruct_y, label_predicted_reconstruct_y = self.discriminator(
                reconstruct_y)
            disc_same_y, label_predicted_same_y = self.discriminator(same_y)

            # Generator adverserial loss
            gen_fake_disc_loss_x = self.generator_loss_deceive_discriminator(
                disc_fake_x)
            gen_reconstruct_disc_loss_x = self.generator_loss_deceive_discriminator(
                disc_reconstruct_x)
            gen_same_disc_loss_x = self.generator_loss_deceive_discriminator(
                disc_same_x)

            gen_fake_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_fake_x)
            gen_reconstruct_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_reconstruct_x)
            gen_same_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_same_x)

            gen_fake_disc_loss_y = self.generator_loss_deceive_discriminator(
                disc_fake_y)
            gen_reconstruct_disc_loss_y = self.generator_loss_deceive_discriminator(
                disc_reconstruct_y)
            gen_same_disc_loss_y = self.generator_loss_deceive_discriminator(
                disc_same_y)

            gen_fake_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_fake_y)
            gen_reconstruct_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_reconstruct_y)
            gen_same_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_same_y)

            # Generator image loss
            reconstruct_image_loss_x = self.reconstruct_loss_fn(
                real_x, reconstruct_x) * self.lambda_reconstruct
            reconstruct_image_loss_y = self.reconstruct_loss_fn(
                real_y, reconstruct_y) * self.lambda_reconstruct

            same_image_loss_x = self.reconstruct_loss_fn(
                real_x, same_x) * self.lambda_reconstruct * self.lambda_identity
            same_image_loss_y = self.reconstruct_loss_fn(
                real_y, same_y) * self.lambda_reconstruct * self.lambda_identity

            # Total generator loss
            gen_total_loss_x = gen_fake_disc_loss_x + gen_reconstruct_disc_loss_x + \
                gen_same_disc_loss_x + \
                reconstruct_image_loss_x + same_image_loss_x + \
                gen_fake_class_loss_x + gen_reconstruct_class_loss_x + gen_same_class_loss_x

            gen_total_loss_y = gen_fake_disc_loss_y + gen_reconstruct_disc_loss_y + \
                gen_same_disc_loss_y + \
                reconstruct_image_loss_y + same_image_loss_y + \
                gen_fake_class_loss_y + gen_reconstruct_class_loss_y + gen_same_class_loss_y

            gen_total_loss = gen_total_loss_x + gen_total_loss_y

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        if self.active_gradient_clip is True:
            gen_grads = adaptive_gradient_clipping(
                gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_generator_loss_x": gen_total_loss_x,
            "total_generator_loss_y": gen_total_loss_y,
            "D_loss_x": disc_fake_loss_x,
            "D_loss_y": disc_fake_loss_y,
            "generator_loss_x": gen_fake_disc_loss_x,
            "generator_loss_y": gen_fake_disc_loss_y,
            "reconstruct_loss_x": reconstruct_image_loss_x,
            "reconstruct_loss_y": reconstruct_image_loss_y,
            "identity_loss_x": same_image_loss_x,
            "identity_loss_y": same_image_loss_y
        }