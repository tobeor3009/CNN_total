import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()


# Define the loss function for the generators
def base_generator_loss_deceive_discriminator(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def base_discriminator_loss_arrest_generator(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

# Define Base Model


class CycleGan(Model):
    def __init__(
        self,
        generator_G=None,
        generator_F=None,
        discriminator_X=None,
        discriminator_Y=None,
        lambda_cycle=10.0,
        lambda_identity=0.5,
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
        self.turn_on_identity_loss = True
        self.turn_on_discriminator_on_identity = False

    def compile(
        self,
        generator_G_optimizer,
        generator_F_optimizer,
        discriminator_X_optimizer,
        discriminator_Y_optimizer,
        image_loss_fn=base_image_loss_fn,
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
        self.cycle_loss_fn = image_loss_fn
        self.identity_loss_fn = image_loss_fn

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
                # Identity mapping
                same_x = self.generator_F(real_x, training=True)
                same_y = self.generator_G(real_y, training=True)

                identity_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
                identity_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
                )
                if self.turn_on_discriminator_on_identity:
                    disc_same_x = self.discriminator_X(same_x, training=True)
                    disc_same_y = self.discriminator_Y(same_y, training=True)

                    generator_G_identity_loss = self.generator_loss_deceive_discriminator(
                        disc_same_y)
                    generator_F_identity_loss = self.generator_loss_deceive_discriminator(
                        disc_same_x)
                    discriminator_X_identity_loss = self.discriminator_loss_arrest_generator(
                        disc_real_x, disc_same_x)
                    discriminator_Y_identity_loss = self.discriminator_loss_arrest_generator(
                        disc_real_y, disc_same_y)
                else:
                    generator_G_identity_loss = 0
                    generator_F_identity_loss = 0
                    discriminator_X_identity_loss = 0
                    discriminator_Y_identity_loss = 0
            else:
                identity_loss_G = 0
                identity_loss_F = 0
                generator_G_identity_loss = 0
                generator_F_identity_loss = 0
                discriminator_X_identity_loss = 0
                discriminator_Y_identity_loss = 0
            # Total generator loss
            total_loss_G = generator_G_loss + cycle_loss_G + \
                generator_G_identity_loss + identity_loss_G
            total_loss_F = generator_F_loss + cycle_loss_F + \
                generator_F_identity_loss + identity_loss_F

            # Discriminator loss
            discriminator_X_loss = self.discriminator_loss_arrest_generator(
                disc_real_x, disc_fake_x)
            discriminator_Y_loss = self.discriminator_loss_arrest_generator(
                disc_real_y, disc_fake_y)

            total_discriminator_X_loss = discriminator_X_loss + \
                discriminator_X_identity_loss * 0.5
            total_discriminator_Y_loss = discriminator_Y_loss + \
                discriminator_Y_identity_loss * 0.5

        # Get the gradients for the generators
        grads_G = tape.gradient(
            total_loss_G, self.generator_G.trainable_variables)
        grads_F = tape.gradient(
            total_loss_F, self.generator_F.trainable_variables)

        # Get the gradients for the discriminators
        discriminator_X_grads = tape.gradient(
            total_discriminator_X_loss, self.discriminator_X.trainable_variables)
        discriminator_Y_grads = tape.gradient(
            total_discriminator_Y_loss, self.discriminator_Y.trainable_variables)

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
            "D_X_loss": total_discriminator_X_loss,
            "D_Y_loss": total_discriminator_Y_loss,
            "generator_G_loss": generator_G_loss,
            "generator_F_loss": generator_F_loss,
            "identity_loss_G": identity_loss_G,
            "identity_loss_F": identity_loss_F,
            "cycle_loss_G": cycle_loss_G,
            "cycle_loss_F": cycle_loss_F
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
