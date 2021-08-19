import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.losses import CategoricalCrossentropy

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = CategoricalCrossentropy()


# Define the loss function for the generators
def base_generator_loss_deceive_discriminator(fake_image):
    fake_loss = adv_loss_fn(tf.ones_like(fake_image), fake_image)
    return fake_loss


# Define the loss function for the discriminators
def base_discriminator_loss_arrest_generator(real_image, fake_image):

    real_loss = adv_loss_fn(tf.ones_like(real_image), real_image)
    fake_loss = adv_loss_fn(tf.zeros_like(fake_image), fake_image)

    return (real_loss + fake_loss) * 0.5


class StarGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        lambda_reconstruct=10.0,
        lambda_class=1,
        lambda_identity=0.5,
    ):
        super(StarGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_class = lambda_class
        self.lambda_identity = lambda_identity
        self.turn_on_identity_loss = True

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss_fn=base_image_loss_fn,
        class_loss_fn=base_class_loss_fn,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
    ):
        super(StarGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.reconstruct_loss_fn = image_loss_fn
        self.identity_loss_fn = image_loss_fn
        self.class_loss_fn = class_loss_fn

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    def train_step(self, batch_data):

        real_x, label_x, real_y, label_y = batch_data

        image_shape = real_x.shape
        label_x_shape = label_x.shape
        label_repeated_x = tf.reshape(
            label_x, (label_x_shape[0], 1, 1, label_x_shape[1]))
        label_repeated_x = keras_backend.repeat_elements(
            label_repeated_x, image_shape[1], axis=1)
        label_repeated_x = keras_backend.repeat_elements(
            label_repeated_x, image_shape[1], axis=2)

        label_y_shape = label_y.shape
        label_repeated_y = tf.reshape(
            label_y, (label_y_shape[0], 1, 1, label_y_shape[1]))
        label_repeated_y = keras_backend.repeat_elements(
            label_repeated_y, image_shape[1], axis=1)
        label_repeated_y = keras_backend.repeat_elements(
            label_repeated_y, image_shape[1], axis=2)

        real_x_for_y = keras_backend.concatenate(
            [real_x, label_repeated_y], axis=-1)
        real_y_for_x = keras_backend.concatenate(
            [real_y, label_repeated_x], axis=-1)

        with tf.GradientTape(persistent=True) as tape:

            # another domain mapping
            fake_x = self.generator(real_y_for_x, training=True)
            fake_y = self.generator(real_x_for_y, training=True)

            fake_x_for_y = keras_backend.concatenate(
                [fake_x, label_repeated_y], axis=-1)
            fake_y_for_x = keras_backend.concatenate(
                [fake_y, label_repeated_x], axis=-1)

            # back to original domain mapping
            reconstruct_x = self.generator(fake_y, label_x, training=True)
            reconstruct_y = self.generator(fake_x, label_y, training=True)

            # Identity mapping
            same_x = self.generator(real_x, label_x, training=True)
            same_y = self.generator(real_y, label_y, training=True)

            # Discriminator output
            disc_real_x, label_predicted_real_x = \
                self.discriminator(real_x, training=True)
            disc_fake_x, label_predicted_fake_x = \
                self.discriminator(fake_x, training=True)

            disc_real_y, label_predicted_real_y = \
                self.discriminator(real_y, training=True)
            disc_fake_y, label_predicted_fake_y = \
                self.discriminator(fake_y, training=True)

            disc_same_x, identity_label_predicted_same_x = \
                self.discriminator(same_x, training=True)

            disc_same_y, identity_label_predicted_same_y = \
                self.discriminator(same_y, training=True)

            # Generator adverserial loss
            generator_loss_x = self.generator_loss_deceive_discriminator(
                disc_fake_x)
            generator_loss_y = self.generator_loss_deceive_discriminator(
                disc_fake_y)

            # Generator reconstruct loss
            reconstruct_loss_x = self.reconstruct_loss_fn(
                real_x, reconstruct_x) * self.lambda_reconstruct
            class_loss_x = self.class_loss_fn(real_x, label_predicted_fake_x)

            reconstruct_loss_y = self.reconstruct_loss_fn(
                real_y, reconstruct_y) * self.lambda_reconstruct
            class_loss_y = self.class_loss_fn(real_y, label_predicted_fake_y)

            # Discriminator loss
            discriminator_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            discriminator_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_real_x)

            discriminator_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y, label_y, label_predicted_real_y)
            discriminator_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_real_y)

            # Generator identity loss
            if self.turn_on_identity_loss is True:
                identity_loss_x = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_reconstruct
                    * self.lambda_identity
                )
                identity_loss_y = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_reconstruct
                    * self.lambda_identity
                )
                discriminator_identity_loss_x = self.discriminator_loss_arrest_generator(
                    disc_real_x, disc_same_x)
                discriminator_identity_class_loss_x = self.class_loss_fn(
                    label_x, identity_label_predicted_same_x)

                discriminator_identity_loss_y = self.discriminator_loss_arrest_generator(
                    disc_real_y, disc_same_y)
                discriminator_identity_class_loss_y = self.class_loss_fn(
                    label_y, identity_label_predicted_same_y)

            else:
                identity_loss_x = 0
                identity_loss_y = 0
                discriminator_identity_loss_x = 0
                discriminator_identity_class_loss_x = 0
                discriminator_identity_loss_y = 0
                discriminator_identity_class_loss_y = 0

            # Total generator loss
            total_loss_x = generator_loss_x + \
                reconstruct_loss_x + class_loss_x + identity_loss_x
            total_loss_y = generator_loss_y + \
                reconstruct_loss_y + class_loss_y + identity_loss_y

            total_generator_loss = total_loss_x + total_loss_y

            discriminator_loss = discriminator_loss_x + discriminator_class_loss_x + \
                discriminator_identity_loss_x + discriminator_identity_class_loss_x + \
                discriminator_loss_y + discriminator_class_loss_y + \
                discriminator_identity_loss_y + discriminator_identity_class_loss_y

        # Get the gradients for the generators
        generator_grads = tape.gradient(
            total_generator_loss, self.generator.trainable_variables)

        # Get the gradients for the discriminators
        discriminator_grads = tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(generator_grads, self.generator.trainable_variables)
        )

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_grads, self.discriminator.trainable_variables)
        )

        return {
            "total_loss_x": total_loss_x,
            "total_loss_y": total_loss_y,
            "D_loss_x": discriminator_loss_x,
            "D_loss_y": discriminator_loss_y,
            "generator_loss_x": generator_loss_x,
            "generator_loss_y": generator_loss_y,
            "identity_loss_x": identity_loss_x,
            "identity_loss_y": identity_loss_y,
            "reconstruct_loss_x": reconstruct_loss_x,
            "reconstruct_loss_y": reconstruct_loss_y
        }
