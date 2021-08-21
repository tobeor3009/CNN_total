import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal
import numpy as np


# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)

# Define the loss function for the generators


def base_generator_loss_deceive_discriminator(fake_img):
    return -tf.reduce_mean(fake_img)


# Define the loss function for the discriminators
def base_discriminator_loss_arrest_generator(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

# Define Base Model


class DCGAN(Model):
    def __init__(self,
                 generator=None,
                 discriminator=None,
                 latent_dim=10,
                 gp_weight=10.0):
        super(DCGAN, self).__init__()

        start_image_shape = (8, 8)
        input_img_size = (256, 256, 3)
        filters = 64
        num_residual_blocks = 9
        num_downsampling = 3

        if generator is None:
            self.generator = get_resnet_generator(latent_dim=latent_dim,
                                                  start_image_shape=start_image_shape,
                                                  output_image_shape=input_img_size,
                                                  filters=filters,
                                                  num_residual_blocks=num_residual_blocks,
                                                  gamma_initializer=gamma_init,
                                                  name="generator")
        else:
            self.generator = generator

        if discriminator is None:
            self.discriminator = get_discriminator(input_img_size=input_img_size,
                                                   num_downsampling=num_downsampling,
                                                   name="discriminator")
        else:
            self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self,
                g_optimizer,
                d_optimizer):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.generator_loss_deceive_discriminator = base_generator_loss_deceive_discriminator
        self.discriminator_loss_arrest_generator = base_discriminator_loss_arrest_generator
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

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

    @tf.function
    def train_step(self, batch):

        real_images, real_target_images = batch

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))
        random_latent_vectors = keras_backend.expand_dims(
            random_latent_vectors, axis=-1)

        # Train the discriminator
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:

            # Decode them to fake images
            fake_images = self.generator(random_latent_vectors, training=True)

            disc_real_x = self.discriminator(real_images, training=True)
            disc_fake_x = self.discriminator(fake_images, training=True)

            g_loss = self.generator_loss_deceive_discriminator(fake_images)

            d_loss = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            d_loss_gradient_panalty = self.gradient_penalty(
                self.discriminator, batch_size, real_images, fake_images)
            d_loss += self.gp_weight * d_loss_gradient_panalty

        disc_grads = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables)

        self.d_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        gen_grads = gen_tape.gradient(g_loss,
                                      self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables))

        # Update metrics
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        return {
            "g_loss": self.g_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
        }


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(tensor=input_tensor,
                      paddings=padding_tensor,
                      mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_generator(
    latent_dim,
    start_image_shape,
    output_image_shape,
    filters=64,
    num_residual_blocks=9,
    gamma_initializer=gamma_init,
    name=None,
):

    upsample_size = output_image_shape[0] // start_image_shape[0]
    num_upsample_blocks = 0
    while upsample_size != 1:
        upsample_size //= 2
        num_upsample_blocks += 1

    img_input = layers.Input(shape=(latent_dim,), name=name + "_img_input")

    x = layers.Dense(np.prod(start_image_shape[:2]) * filters)(img_input)
    x = layers.Reshape((*start_image_shape[:2], filters))(x)
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = layers.LeakyReLU(0.2)(x)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.LeakyReLU(0.2))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.LeakyReLU(0.2))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = Model(img_input, x, name=name)
    return model


def get_discriminator(
    input_img_size,
    filters=64,
    kernel_initializer=kernel_init,
    num_downsampling=3,
    name=None
):
    DROPOUT_RATIO = 0.5

    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(num_downsampling):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    # (Batch_Size,?)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATIO)(x)
    # let's add a fully-connected layer
    # (Batch_Size,1)
    x = layers.Dense(1024, activation='relu')(x)
    # (Batch_Size,1024)
    x = layers.Dropout(DROPOUT_RATIO)(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=img_input, outputs=predictions, name=name)
    return model
