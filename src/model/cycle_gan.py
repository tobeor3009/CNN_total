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


class CycleGan(Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.generator_G = generator_G
        self.generator_F = generator_F
        self.discriminator_X = discriminator_X
        self.discriminator_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.turn_on_identity_loss = True

    def compile(
        self,
        generator_G_optimizer,
        generator_F_optimizer,
        discriminator_X_optimizer,
        discriminator_Y_optimizer,
        image_loss,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
    ):
        super(CycleGan, self).compile()
        self.generator_G_optimizer = generator_G_optimizer
        self.generator_F_optimizer = generator_F_optimizer
        self.discriminator_X_optimizer = discriminator_X_optimizer
        self.discriminator_Y_optimizer = discriminator_Y_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.cycle_loss_fn = image_loss
        self.identity_loss_fn = image_loss

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    def train_step(self, batch_data):

        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:

            # another domain mapping
            fake_x = self.generator_F(real_y, training=True)
            fake_y = self.generator_G(real_x, training=True)

            # back to original domain mapping
            cycled_x = self.generator_F(fake_y, training=True)
            cycled_y = self.generator_G(fake_x, training=True)

            # Identity mapping
            same_x = self.generator_F(real_x, training=True)
            same_y = self.generator_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.discriminator_X(real_x, training=True)
            disc_fake_x = self.discriminator_X(fake_x, training=True)

            disc_real_y = self.discriminator_Y(real_y, training=True)
            disc_fake_y = self.discriminator_Y(fake_y, training=True)

            # Generator adverserial loss
            generator_G_loss = self.generator_loss_deceive_discriminator(
                disc_fake_y)
            generator_F_loss = self.generator_loss_deceive_discriminator(
                disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(
                real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(
                real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            if self.turn_on_identity_loss is True:
                id_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
                id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
            else:
                id_loss_G = 0
                id_loss_F = 0
            # Total generator loss
            total_loss_G = generator_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = generator_F_loss + cycle_loss_F + id_loss_F
            # Discriminator loss
            discriminator_X_loss = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            discriminator_Y_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(
            total_loss_G, self.generator_G.trainable_variables)
        grads_F = tape.gradient(
            total_loss_F, self.generator_F.trainable_variables)

        # Get the gradients for the discriminators
        discriminator_X_grads = tape.gradient(
            discriminator_X_loss, self.discriminator_X.trainable_variables)
        discriminator_Y_grads = tape.gradient(
            discriminator_Y_loss, self.discriminator_Y.trainable_variables)

        # Update the weights of the generators
        self.generator_G_optimizer.apply_gradients(
            zip(grads_G, self.generator_G.trainable_variables)
        )
        self.generator_F_optimizer.apply_gradients(
            zip(grads_F, self.generator_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.discriminator_X_optimizer.apply_gradients(
            zip(discriminator_X_grads, self.discriminator_X.trainable_variables)
        )
        self.discriminator_Y_optimizer.apply_gradients(
            zip(discriminator_Y_grads, self.discriminator_Y.trainable_variables)
        )

        return {
            "total_loss_G": total_loss_G,
            "total_loss_F": total_loss_F,
            "D_X_loss": discriminator_X_loss,
            "D_Y_loss": discriminator_Y_loss,
            "generator_G_loss": generator_G_loss,
            "generator_F_loss": generator_F_loss,
            "id_loss_G": id_loss_G,
            "id_loss_F": id_loss_F,
            "cycle_loss_G": cycle_loss_G,
            "cycle_loss_F": cycle_loss_F
        }
