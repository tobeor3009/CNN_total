import tensorflow as tf
from tensorflow.keras import layers, backend
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import BinaryCrossentropy

from ..util.lsgan import to_real_loss, to_fake_loss
from ..util.grad_clip import adaptive_gradient_clipping
from ..inception_resnet_v2.util.pathology import recon_overlapping_patches_quarter_scale

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = BinaryCrossentropy(label_smoothing=0.01)
mean_layer = layers.Average()


def sqrt_with_sign(x):
    """
    Compute square root of input tensor while preserving its sign.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Tensor of same shape as input with square root of its absolute
                   values and original signs.
    """
    return tf.sign(x) * tf.sqrt(tf.abs(x))


def tile_concat_2d(a, b, image_shape):
    b = b[:, None, None, :]
    b = tf.tile(b, [1, image_shape[0], image_shape[1], 1])
    return tf.concat([a, b], axis=-1)


def get_diff_label(label_tensor):
    label_shape = tf.shape(label_tensor)
    batch_size, num_class = label_shape[0], label_shape[1]
    label_tensor = backend.random_normal(
        shape=(batch_size, num_class)) - label_tensor
    label_tensor = tf.argmax(label_tensor, axis=1)
    label_tensor = tf.one_hot(label_tensor, depth=num_class)
    return label_tensor


def interpolate(label_shape, real, fake, mode="2d"):
    alpha = tf.random.normal([label_shape[0]], 0.0, 1.0)
    if mode == "2d":
        image_alpha = alpha[:, None, None, None]
    elif mode == "3d":
        image_alpha = alpha[:, None, None, None, None]
    inter = real + image_alpha * (fake - real)
    return inter


@tf.function
def gradient_penalty(disc, real, fake,
                     mode="2d", smooth=1e-7):
    label_shape = tf.shape(real)
    inter = interpolate(label_shape, real, fake,
                        mode=mode)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inter)
        # 1. Get the discriminator output for this interpolated image.
        # disc.output = [validity, class]
        pred, _ = disc(inter, training=True)
    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [inter])[0]
    # grads = tf.reshape(grads, [label_shape[0], -1])
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads + smooth), axis=[1, 2, 3]))
    gp = (norm - 1.0) ** 2
    # gp = tf.reduce_mean((norm - 1.0) ** 2)
    # 3. Calculate the norm of the gradients.
    return gp


class ATTGan(Model):
    def __init__(
        self,
        generator,
        discriminator
    ):
        super(ATTGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        _, self.image_h, self.image_w, _ = self.generator.output.shape
        _, self.feature_h, self.feature_w, _ = self.discriminator.output[0].shape

    def compile(
        self,
        batch_size,
        image_shape,
        generator_optimizer,
        discriminator_optimizer,
        image_loss_fn=base_image_loss_fn,
        class_loss_fn=base_class_loss_fn,
        disc_loss_fn=None,
        disc_real_loss_fn=to_real_loss,
        disc_fake_loss_fn=to_fake_loss,
        active_gradient_clip=True,
        image_loss_coef=100.0,
        gen_fake_y_ratio=3.0,
        gen_disc_loss_coef=1.0,
        gen_class_loss_coef=0.5,
        disc_class_loss_coef=0.5,
        lambda_clip=0.1
    ):
        super(ATTGan, self).compile()
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.recon_loss_fn = image_loss_fn
        self.class_loss_fn = class_loss_fn
        self.active_gradient_clip = active_gradient_clip
        self.lambda_clip = lambda_clip
        self.image_loss_coef = image_loss_coef
        self.disc_loss_fn = disc_loss_fn
        self.disc_real_loss_fn = disc_real_loss_fn
        self.disc_fake_loss_fn = disc_fake_loss_fn
        self.gen_fake_y_ratio = gen_fake_y_ratio
        self.gen_disc_loss_coef = gen_disc_loss_coef
        self.gen_class_loss_coef = gen_class_loss_coef
        self.disc_class_loss_coef = disc_class_loss_coef

    # not in use. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    # inference When random cropping
    def patch_call(self, inputs, include_disc=True):
        image_array, label_array = inputs
        image_array = tf.stop_gradient(image_array)
        label_array = tf.stop_gradient(label_array)

        image_shape = tf.shape(image_array)
        H, W = image_shape[1], image_shape[2]

        h_start_point_list = [H // 8 * idx for idx in range(7)]
        h_end_point_list = [H // 8 * idx for idx in range(2, 9)]
        w_start_point_list = [W // 8 * idx for idx in range(7)]
        w_end_point_list = [W // 8 * idx for idx in range(2, 9)]
        image_partial_list = []
        validity_list = []
        pred_label_list = []
        for h_start, h_end in zip(h_start_point_list, h_end_point_list):
            for w_start, w_end in zip(w_start_point_list, w_end_point_list):
                image_partial = image_array[:, h_start:h_end, w_start:w_end]
                image_partial = self.generator([image_partial, label_array])
                image_partial_list.append(image_partial)
                if include_disc:
                    validity, pred_label = self.discriminator(image_partial)
                    validity_list.append(validity)
                    pred_label_list.append(pred_label)
        pred_image_array = recon_overlapping_patches_quarter_scale(
            image_partial_list)
        if include_disc:
            validity = recon_overlapping_patches_quarter_scale(validity_list)
            pred_label = mean_layer(pred_label_list)
            return pred_image_array, validity, pred_label
        else:
            return pred_image_array

    # Remark this line because it conflict with multi gpu call such as strategy.scope()
    # @tf.function
    def train_step(self, batch_data):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, label_x = batch_data
        batch_size = tf.shape(label_x)[0]
        label_y = get_diff_label(label_x)
        same_label = label_x - label_x
        target_label = label_y - label_x
        feature_alpha = tf.random.uniform((batch_size, self.feature_h, self.feature_w, 1),
                                          0, 1)
        feature_threshold = tf.random.uniform((batch_size, 1, 1, 1),
                                              0, 1)
        feature_alpha = tf.cast(feature_alpha >= feature_threshold,
                                tf.float32)

        image_alpha = tf.image.resize(feature_alpha, (self.image_h, self.image_w),
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:
            # another domain mapping
            fake_y = self.generator([real_x, label_y])
            # Discriminator output
            blend_image = (real_x * image_alpha +
                           fake_y * (1 - image_alpha))
            # disc_blend, _ = self.discriminator(blend_image,
            #                                 training=True)
            disc_real, label_real_x = self.discriminator(real_x,
                                                         training=True)
            disc_fake, _ = self.discriminator(fake_y,
                                              training=True)
            label_real_x_loss = backend.mean(self.class_loss_fn(label_x,
                                                                label_real_x))
            # disc_blend_loss = backend.mean(self.disc_loss_fn(feature_alpha,
            #                                                  disc_blend))
            disc_real_loss = backend.mean(self.disc_real_loss_fn(disc_real))
            disc_fake_loss = backend.mean(self.disc_fake_loss_fn(disc_fake))
            disc_gp = gradient_penalty(self.discriminator, real_x, fake_y,
                                       mode="2d")
            product_loss = sqrt_with_sign(-disc_real_loss * disc_fake_loss)

            class_coef = tf.math.maximum(tf.stop_gradient(product_loss),
                                         self.disc_class_loss_coef)
            sum_loss = disc_real_loss + disc_fake_loss
            disc_total_loss = (sum_loss + disc_gp * 1.0 +
                               label_real_x_loss * class_coef)
        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(disc_total_loss,
                                        self.discriminator.trainable_variables)
        if self.active_gradient_clip is True:
            disc_grads = adaptive_gradient_clipping(disc_grads,
                                                    self.discriminator.trainable_variables,
                                                    lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=False) as gen_tape:
            # another domain mapping
            fake_y = self.generator([real_x, label_y],
                                    training=True)
            recon_x = self.generator([real_x, label_x],
                                     training=True)
            # Discriminator output
            disc_fake_y, label_fake_y = self.discriminator(fake_y)
            label_fake_y_loss = backend.mean(self.class_loss_fn(label_y,
                                                                label_fake_y))
            gen_label_loss = label_fake_y_loss

            # Generator adverserial loss
            gen_fake_y_disc_loss = backend.mean(
                self.disc_real_loss_fn(disc_fake_y))

            gen_disc_loss = gen_fake_y_disc_loss
            # Generator image loss
            recon_x_image_loss = self.recon_loss_fn(real_x,
                                                    recon_x)
            gen_image_loss = backend.mean(recon_x_image_loss)
            # Total generator loss
            gen_total_loss = (gen_disc_loss * self.gen_disc_loss_coef +
                              gen_image_loss * self.image_loss_coef +
                              gen_label_loss * self.gen_class_loss_coef)

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(gen_total_loss,
                                      self.generator.trainable_variables)
        if self.active_gradient_clip is True:
            gen_grads = adaptive_gradient_clipping(gen_grads,
                                                   self.generator.trainable_variables,
                                                   lambda_clip=self.lambda_clip)
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        return {
            "total_disc_loss": disc_total_loss,
            "disc_real_loss": disc_real_loss,
            "disc_fake_loss": disc_fake_loss,
            "disc_gp": disc_gp,
            "label_real_x_loss": label_real_x_loss,
            "total_gen_loss": gen_total_loss,
            "gen_fake_y_disc_loss": gen_fake_y_disc_loss,
            "gen_class_loss": gen_label_loss,
            "recon_x_image_loss": recon_x_image_loss,
        }
