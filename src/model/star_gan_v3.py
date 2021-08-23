import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.losses import CategoricalCrossentropy

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = CategoricalCrossentropy(label_smoothing=0.01)


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


class StarGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        lambda_reconstruct=10.0,
        lambda_identity=0.5,
        gp_weight=10.0
    ):
        super(StarGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
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

        label_repeated_x = layers.Reshape(
            (1, 1, self.label_num))(label_x)
        label_repeated_x = keras_backend.repeat_elements(
            label_repeated_x, self.image_shape[0], axis=1)
        label_repeated_x = keras_backend.repeat_elements(
            label_repeated_x, self.image_shape[1], axis=2)

        label_repeated_y = layers.Reshape(
            (1, 1, self.label_num))(label_y)
        label_repeated_y = keras_backend.repeat_elements(
            label_repeated_y, self.image_shape[0], axis=1)
        label_repeated_y = keras_backend.repeat_elements(
            label_repeated_y, self.image_shape[1], axis=2)

        real_x_for_y = keras_backend.concatenate(
            [real_x, label_repeated_x, label_repeated_y], axis=-1)
        real_y_for_x = keras_backend.concatenate(
            [real_y, label_repeated_y, label_repeated_x], axis=-1)

        real_x_for_x = keras_backend.concatenate(
            [real_x, label_repeated_x, label_repeated_x], axis=-1)
        real_y_for_y = keras_backend.concatenate(
            [real_x, label_repeated_y, label_repeated_y], axis=-1)

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as disc_tape:

            # another domain mapping
            fake_x = self.generator(real_y_for_x)
            fake_y = self.generator(real_x_for_y)

            fake_x_for_y = keras_backend.concatenate(
                [fake_x, label_repeated_x, label_repeated_y], axis=-1)
            fake_y_for_x = keras_backend.concatenate(
                [fake_y, label_repeated_y, label_repeated_x], axis=-1)

            # back to original domain mapping
            reconstruct_x = self.generator(fake_y_for_x)
            reconstruct_y = self.generator(fake_x_for_y)

            # Identity mapping
            same_x = self.generator(real_x_for_x)
            same_y = self.generator(real_y_for_y)

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

            # Compute Discriminator loss_x
            disc_fake_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_fake_x)
            disc_reconstruct_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_reconstruct_x)
            disc_same_class_loss_x = self.class_loss_fn(
                label_x, label_predicted_same_x)

            disc_fake_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            disc_fake_x_gradient_panalty = self.gradient_penalty(
                self.discriminator, self.batch_size, real_x, fake_x)
            disc_fake_loss_x += self.gp_weight * disc_fake_x_gradient_panalty

            disc_reconstruct_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_reconstruct_x)
            disc_reconstruct_x_gradient_panalty = self.gradient_penalty(
                self.discriminator, self.batch_size, real_x, reconstruct_x)
            disc_reconstruct_loss_x += self.gp_weight * disc_reconstruct_x_gradient_panalty

            disc_identity_loss_x = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_same_x)
            disc_identity_x_gradient_panalty = self.gradient_penalty(
                self.discriminator, self.batch_size, real_x, same_x)
            disc_identity_loss_x += self.gp_weight * disc_identity_x_gradient_panalty
            disc_identity_loss_x *= self.lambda_identity

            disc_class_loss_real_x = self.class_loss_fn(
                label_x, label_predicted_real_x)

            # Compute Discriminator loss_y
            disc_fake_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_fake_y)
            disc_reconstruct_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_reconstruct_y)
            disc_same_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_same_y)

            disc_fake_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)
            disc_fake_y_gradient_panalty = self.gradient_penalty(
                self.discriminator, self.batch_size, real_y, fake_y)
            disc_fake_loss_y += self.gp_weight * disc_fake_y_gradient_panalty

            disc_reconstruct_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_reconstruct_y)
            disc_reconstruct_y_gradient_panalty = self.gradient_penalty(
                self.discriminator, self.batch_size, real_y, reconstruct_y)
            disc_reconstruct_loss_y += self.gp_weight * disc_reconstruct_y_gradient_panalty

            disc_identity_loss_y = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_same_y)
            disc_identity_y_gradient_panalty = self.gradient_penalty(
                self.discriminator, self.batch_size, real_y, same_y)
            disc_identity_loss_y += self.gp_weight * disc_identity_y_gradient_panalty
            disc_identity_loss_y *= self.lambda_identity

            disc_class_loss_real_y = self.class_loss_fn(
                label_y, label_predicted_real_y)

            disc_total_loss_x = disc_fake_loss_x + disc_reconstruct_loss_x + disc_identity_loss_x + \
                disc_class_loss_real_x + disc_same_class_loss_x + \
                disc_fake_class_loss_x + disc_reconstruct_class_loss_x
            disc_total_loss_y = disc_fake_loss_y + disc_reconstruct_loss_y + disc_identity_loss_y + \
                disc_class_loss_real_y + disc_same_class_loss_y + \
                disc_fake_class_loss_y + disc_reconstruct_class_loss_y

            disc_total_loss = disc_total_loss_x + disc_total_loss_y

        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(
            disc_total_loss, self.discriminator.trainable_variables)
        cliped_disc_grads = self.active_gradient_clipping(
            disc_grads, self.discriminator.trainable_variables)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(cliped_disc_grads, self.discriminator.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as gen_tape:
            # another domain mapping
            fake_x = self.generator(real_y_for_x, training=True)
            fake_y = self.generator(real_x_for_y, training=True)

            fake_x_for_y = keras_backend.concatenate(
                [fake_x, label_repeated_x, label_repeated_y], axis=-1)
            fake_y_for_x = keras_backend.concatenate(
                [fake_y, label_repeated_y, label_repeated_x], axis=-1)

            # back to original domain mapping
            reconstruct_x = self.generator(fake_y_for_x, training=True)
            reconstruct_y = self.generator(fake_x_for_y, training=True)

            # Identity mapping
            same_x = self.generator(real_x_for_x, training=True)
            same_y = self.generator(real_y_for_y, training=True)

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
            gen_fake_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_fake_y)
            gen_reconstruct_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_reconstruct_y)
            gen_same_class_loss_y = self.class_loss_fn(
                label_y, label_predicted_same_y)

            # Generator reconstruct loss
            reconstruct_image_loss_x = self.reconstruct_loss_fn(
                real_x, reconstruct_x) * self.lambda_reconstruct
            reconstruct_image_loss_y = self.reconstruct_loss_fn(
                real_y, reconstruct_y) * self.lambda_reconstruct

            gen_identity_image_loss_x = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_reconstruct
                * self.lambda_identity
            )
            gen_identity_image_loss_y = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_reconstruct
                * self.lambda_identity
            )

            # Total generator loss
            gen_total_loss_x = gen_fake_disc_loss_x + gen_reconstruct_disc_loss_x + \
                reconstruct_image_loss_x + gen_identity_image_loss_x + \
                gen_fake_class_loss_x + gen_reconstruct_class_loss_x + \
                gen_same_class_loss_x
            gen_total_loss_y = gen_fake_disc_loss_y + gen_reconstruct_disc_loss_y + \
                reconstruct_image_loss_y + gen_identity_image_loss_y + \
                gen_fake_class_loss_y + gen_reconstruct_class_loss_y + \
                gen_same_class_loss_y

            gen_total_loss = gen_total_loss_x + gen_total_loss_y

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        cliped_gen_grads = self.active_gradient_clipping(
            gen_grads, self.generator.trainable_variables)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(cliped_gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_generator_loss_x": gen_total_loss_x,
            "total_generator_loss_y": gen_total_loss_y,
            "D_loss_x": disc_fake_loss_x,
            "D_loss_y": disc_fake_loss_y,
            "generator_loss_x": gen_fake_disc_loss_x,
            "generator_loss_y": gen_fake_disc_loss_y,
            "identity_loss_x": gen_identity_image_loss_x,
            "identity_loss_y": gen_identity_image_loss_y,
            "reconstruct_loss_x": reconstruct_image_loss_x,
            "reconstruct_loss_y": reconstruct_image_loss_y
        }
