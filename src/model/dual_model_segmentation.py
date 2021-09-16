import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal
import numpy as np

from .util.lsgan import base_generator_loss_deceive_discriminator, base_discriminator_loss_arrest_generator
from .util.grad_clip import adaptive_gradient_clipping
from .util.gan_loss import rgb_color_histogram_loss

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)


class DualModelSegmentation(Model):
    def __init__(self,
                 generator_A=None,
                 generator_B=None):
        super(DualModelSegmentation, self).__init__()

        self.generator_A = generator_A
        self.generator_B = generator_B

    def compile(self,
                generator_A_optimizer,
                generator_B_optimizer,
                image_loss,
                metrics,
                lambda_clip=0.1):
        super(DualModelSegmentation, self).compile()
        self.generator_A_optimizer = generator_A_optimizer
        self.generator_B_optimizer = generator_B_optimizer
        self.image_loss = image_loss
        self.metric = metrics[0]
        self.lambda_clip = lambda_clip

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    @tf.function
    def train_step(self, batch):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_images, mask_images = batch

        gen_A_mask = mask_images[:, :, :, :1]
        gen_B_mask = mask_images[:, :, :, 1:]
        with tf.GradientTape(persistent=True) as gen_tape:
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Decode them to fake images
            gen_A_fake_mask = self.generator_A(real_images, training=True)
            gen_B_fake_mask = self.generator_B(real_images, training=True)

            gen_A_loss = self.image_loss(gen_A_mask, gen_A_fake_mask)
            gen_A_metric = self.metric(gen_A_mask, gen_A_fake_mask)
            gen_B_loss = self.image_loss(gen_B_mask, gen_B_fake_mask)
            gen_B_metric = self.metric(gen_B_mask, gen_B_fake_mask)
        # Get the gradients for the discriminators
        gen_A_grads = gen_tape.gradient(
            gen_A_loss, self.generator_A.trainable_variables)
        gen_B_grads = gen_tape.gradient(
            gen_B_loss, self.generator_B.trainable_variables)

        cliped_gen_A_grads = adaptive_gradient_clipping(
            gen_A_grads, self.generator_A.trainable_variables, lambda_clip=self.lambda_clip)
        cliped_gen_B_grads = adaptive_gradient_clipping(
            gen_B_grads, self.generator_B.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.generator_A_optimizer.apply_gradients(
            zip(cliped_gen_A_grads, self.generator_A.trainable_variables)
        )
        # Update the weights of the discriminators
        self.generator_B_optimizer.apply_gradients(
            zip(cliped_gen_B_grads, self.generator_B.trainable_variables)
        )
        return {
            "gen_A_loss": gen_A_loss,
            "gen_B_loss": gen_B_loss,
            "gen_A_metric": gen_A_metric,
            "gen_B_metric": gen_B_metric,
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
