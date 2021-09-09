import tensorflow as tf
from tensorflow.keras import backend as keras_backend
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal

from .util.gan_loss import rgb_color_histogram_loss
from .util.wgan_gp import base_generator_loss_deceive_discriminator, \
    base_discriminator_loss_arrest_generator, gradient_penalty
from .util.grad_clip import adaptive_gradient_clipping

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class CycleGan(Model):
    def __init__(
        self,
        generator_G=None,
        generator_F=None,
        discriminator_X=None,
        discriminator_Y=None,
        lambda_cycle=10.0,
        lambda_identity=0.5,
        lambda_histogram=0.,
        gp_weight=10.0
    ):
        super(CycleGan, self).__init__()

        input_img_size = (256, 256, 3)
        filters = 64
        num_downsampling_blocks = 2
        num_residual_blocks = 9
        num_upsample_blocks = 2
        num_downsampling = 3

        if generator_G is None:
            self.generator_G = get_resnet_generator(input_img_size=input_img_size,
                                                    filters=filters,
                                                    num_downsampling_blocks=num_downsampling_blocks,
                                                    num_residual_blocks=num_residual_blocks,
                                                    num_upsample_blocks=num_upsample_blocks,
                                                    name="generator_G")
        else:
            self.generator_G = generator_G

        if generator_F is None:
            self.generator_F = get_resnet_generator(input_img_size=input_img_size,
                                                    filters=filters,
                                                    num_downsampling_blocks=num_downsampling_blocks,
                                                    num_residual_blocks=num_residual_blocks,
                                                    num_upsample_blocks=num_upsample_blocks,
                                                    name="generator_F")
        else:
            self.generator_F = generator_F

        if discriminator_X is None:
            self.discriminator_X = get_discriminator(input_img_size=input_img_size,
                                                     num_downsampling=num_downsampling,
                                                     name="discriminator_X")
        else:
            self.discriminator_X = discriminator_X

        if discriminator_Y is None:
            self.discriminator_Y = get_discriminator(input_img_size=input_img_size,
                                                     num_downsampling=num_downsampling,
                                                     name="discriminator_Y")
        else:
            self.discriminator_Y = discriminator_Y

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_histogram = lambda_histogram
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
        histogram_loss=True,
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
        self.histogram_loss = histogram_loss
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
        real_x, real_y = batch_data
        batch_size = tf.shape(real_x)[0]

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as disc_tape:
            # another domain mapping
            fake_x = self.generator_F(real_y)
            fake_y = self.generator_G(real_x)

            # back to original domain mapping
            cycle_x = self.generator_F(fake_y)
            cycle_y = self.generator_G(fake_x)

            # Discriminator output
            disc_real_x = self.discriminator_X(real_x, training=True)
            disc_fake_x = self.discriminator_X(fake_x, training=True)
            disc_cycle_x = self.discriminator_X(cycle_x, training=True)

            disc_real_y = self.discriminator_Y(real_y, training=True)
            disc_fake_y = self.discriminator_Y(fake_y, training=True)
            disc_cycle_y = self.discriminator_Y(cycle_y, training=True)

            if self.identity_loss is True:
                # Identity mapping
                same_x = self.generator_F(real_x)
                same_y = self.generator_G(real_y)

                disc_same_x = self.discriminator_X(same_x, training=True)
                disc_same_y = self.discriminator_Y(same_y, training=True)

                disc_X_identity_loss = self.discriminator_loss_arrest_generator(
                    disc_real_x, disc_same_x)
                disc_Y_identity_loss = self.discriminator_loss_arrest_generator(
                    disc_real_y, disc_same_y)

                disc_X_identity_gradient_panalty = gradient_penalty(
                    self.discriminator_X, batch_size, real_x, same_x)
                disc_Y_identity_gradient_panalty = gradient_penalty(
                    self.discriminator_Y, batch_size, real_y, same_y)

                disc_X_identity_loss += self.gp_weight * \
                    disc_X_identity_gradient_panalty
                disc_Y_identity_loss += self.gp_weight * \
                    disc_Y_identity_gradient_panalty
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

            disc_X_fake_gradient_panalty = gradient_penalty(
                self.discriminator_X, batch_size, real_x, fake_x)
            disc_Y_fake_gradient_panalty = gradient_penalty(
                self.discriminator_Y, batch_size, real_y, fake_y)
            disc_X_cycle_gradient_panalty = gradient_penalty(
                self.discriminator_X, batch_size, real_x, cycle_x)
            disc_Y_cycle_gradient_panalty = gradient_penalty(
                self.discriminator_Y, batch_size, real_y, cycle_y)

            disc_X_fake_loss += self.gp_weight * disc_X_fake_gradient_panalty
            disc_Y_fake_loss += self.gp_weight * disc_Y_fake_gradient_panalty
            disc_X_cycle_loss += self.gp_weight * disc_X_cycle_gradient_panalty
            disc_Y_cycle_loss += self.gp_weight * disc_Y_cycle_gradient_panalty

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
            fake_x = self.generator_F(real_y, training=True)
            fake_y = self.generator_G(real_x, training=True)

            # back to original domain mapping
            cycle_x = self.generator_F(fake_y, training=True)
            cycle_y = self.generator_G(fake_x, training=True)

            if self.histogram_loss is True:
                gen_G_histo_loss = self.lambda_histogram * \
                    rgb_color_histogram_loss(real_y, fake_y)
                gen_F_histo_loss = self.lambda_histogram * \
                    rgb_color_histogram_loss(real_x, fake_x)

            # Discriminator output
            disc_fake_x = self.discriminator_X(fake_x)
            disc_fake_y = self.discriminator_Y(fake_y)

            disc_cycle_x = self.discriminator_X(cycle_x)
            disc_cycle_y = self.discriminator_Y(cycle_y)

            if self.identity_loss is True:
                # Identity mapping
                same_x = self.generator_F(real_x, training=True)
                same_y = self.generator_G(real_y, training=True)

                disc_same_x = self.discriminator_X(same_x)
                disc_same_y = self.discriminator_Y(same_y)

                gen_G_identity_image_loss = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
                gen_F_identity_image_loss = (
                    self.identity_loss_fn(real_x, same_x)
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

            # Total generator loss
            gen_G_total_loss = gen_G_fake_disc_loss + \
                gen_G_cycle_image_loss + gen_G_cycle_disc_loss + \
                gen_G_identity_image_loss + gen_G_identity_disc_loss
            gen_F_total_loss = gen_F_fake_disc_loss + \
                gen_F_cycle_image_loss + gen_F_cycle_disc_loss + \
                gen_F_identity_image_loss + gen_F_identity_disc_loss

            if self.histogram_loss is True:
                gen_G_total_loss += gen_G_histo_loss
                gen_F_total_loss += gen_F_histo_loss
            else:
                gen_G_histo_loss = 0
                gen_F_histo_loss = 0
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
            "generator_G_histo_loss": gen_G_histo_loss,
            "generator_F_histo_loss": gen_F_histo_loss,
            "identity_loss_G": gen_G_identity_image_loss,
            "identity_loss_F": gen_F_identity_image_loss,
            "cycle_loss_G": gen_G_cycle_image_loss,
            "cycle_loss_F": gen_F_cycle_image_loss
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
    # x = tfa.layers.InstanceNormalization(
    #     gamma_initializer=gamma_initializer)(x)
    x = layers.LayerNormalization(
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
    input_img_size,
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters,
                       activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

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

    model = Model(inputs=img_input, outputs=x, name=name)
    return model
