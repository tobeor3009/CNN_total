import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
# Define the loss function for the generators


def base_generator_loss_deceive_discriminator(fake_img):
    return -tf.reduce_mean(fake_img)


# Define the loss function for the discriminators
def base_discriminator_loss_arrest_generator(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def compute_l2_norm(tensor):

    squared = keras_backend.square(tensor)
    l2_norm = keras_backend.sum(squared)
    l2_norm = keras_backend.sqrt(l2_norm)

    return l2_norm


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
        image_loss=base_image_loss_fn,
        generator_against_discriminator=base_generator_loss_deceive_discriminator,
        discriminator_loss_arrest_generator=base_discriminator_loss_arrest_generator,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_deceive_discriminator = generator_against_discriminator
        self.discriminator_loss_arrest_generator = discriminator_loss_arrest_generator
        self.image_loss = image_loss
        self.lambda_clip = lambda_clip
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    def gradient_penalty(self, discriminator, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def active_gradient_clipping(self, grad_list, trainable_variable_list):

        cliped_grad_list = []

        for grad, trainable_variable in zip(grad_list, trainable_variable_list):
            grad_l2_norm = compute_l2_norm(grad)
            trainable_variable_l2_norm = compute_l2_norm(trainable_variable)

            clip_value = self.lambda_clip * \
                (trainable_variable_l2_norm / grad_l2_norm)
            cliped_grad = keras_backend.clip(grad, -clip_value, clip_value)

            cliped_grad_list.append(cliped_grad)

        return cliped_grad_list

    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, real_y = batch_data

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator(real_x)

            # Discriminator output
            disc_real_y = self.discriminator(real_y, training=True)
            disc_fake_y = self.discriminator(fake_y, training=True)

            # Discriminator loss
            disc_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)

        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        cliped_disc_grads = self.active_gradient_clipping(
            disc_grads, self.discriminator.trainable_variables)

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

            # Discriminator output
            disc_fake_y = self.discriminator(fake_y)
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            gen_loss_adverserial_loss = self.generator_loss_deceive_discriminator(
                disc_fake_y)
            total_generator_loss = gen_loss_in_real_y + gen_loss_adverserial_loss

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(
            total_generator_loss, self.generator.trainable_variables)
        cliped_gen_grads = self.active_gradient_clipping(
            gen_grads, self.generator.trainable_variables)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(cliped_gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_generator_loss": total_generator_loss,
            "generator_loss_in_real_y": gen_loss_in_real_y,
            "generator_loss_adverserial_loss": gen_loss_adverserial_loss,
            "discriminator_loss_loss": disc_loss,
        }
