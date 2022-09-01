import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()

# Define the loss function for the generators


def base_generator_loss_deceive_discriminator(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def base_discriminator_loss_arrest_generator(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


class Pix2PixGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.image_loss = image_loss

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:

            # another domain mapping
            fake_y = self.generator(real_x, training=True)

            # Discriminator output
            disc_real_y = self.discriminator(real_y, training=True)
            disc_fake_y = self.discriminator(fake_y, training=True)

            # Generator paired real y loss
            generator_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            generator_loss_adverserial_loss = self.generator_loss_deceive_discriminator(
                disc_fake_y)
            total_generator_loss = generator_loss_in_real_y + generator_loss_adverserial_loss

            # Discriminator loss
            discriminator_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)

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
            "total_generator_loss": total_generator_loss,
            "generator_loss_in_real_y": generator_loss_in_real_y,
            "generator_loss_adverserial_loss": generator_loss_adverserial_loss,
            "discriminator_loss_loss": discriminator_loss,
        }
