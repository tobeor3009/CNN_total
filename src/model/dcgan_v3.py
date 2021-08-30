import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal
import numpy as np

from .util.lsgan import base_generator_loss_deceive_discriminator, base_discriminator_loss_arrest_generator
from .util.grad_clip import active_gradient_clipping


# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)


class DCGAN(Model):
    def __init__(self,
                 generator=None,
                 discriminator=None,
                 latent_dim=10):
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

    def compile(self,
                g_optimizer,
                d_optimizer,
                image_loss=base_image_loss_fn,
                lambda_clip=0.1):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.image_loss = image_loss
        self.generator_loss_deceive_discriminator = base_generator_loss_deceive_discriminator
        self.discriminator_loss_arrest_generator = base_discriminator_loss_arrest_generator
        self.lambda_clip = lambda_clip

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    @tf.function
    def train_step(self, batch):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_images, real_target_images = batch

        with tf.GradientTape(persistent=False) as disc_tape:
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Decode them to fake images
            fake_images = self.generator(real_images)

            disc_real_x = self.discriminator(real_images, training=True)
            disc_fake_x = self.discriminator(fake_images, training=True)

            disc_loss = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        cliped_disc_grads = active_gradient_clipping(
            disc_grads, self.discriminator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.d_optimizer.apply_gradients(
            zip(cliped_disc_grads, self.discriminator.trainable_variables)
        )
        with tf.GradientTape(persistent=False) as gen_tape:
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            fake_images = self.generator(real_images, training=True)
            disc_fake_x = self.discriminator(fake_images, training=True)

            gen_image_loss = self.image_loss(real_images, fake_images)
            gen_disc_loss = self.generator_loss_deceive_discriminator(
                fake_images)

            gen_loss = gen_image_loss + gen_disc_loss
        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(gen_loss,
                                      self.generator.trainable_variables)
        cliped_gen_grads = active_gradient_clipping(
            gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.g_optimizer.apply_gradients(
            zip(cliped_gen_grads, self.generator.trainable_variables))

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
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
