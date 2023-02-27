import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, backend
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

from .util.lsgan import to_real_loss, to_fake_loss
from .util.grad_clip import adaptive_gradient_clipping
from .inception_resnet_v2.util.pathology import recon_overlapping_patches_quarter_scale

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = BinaryCrossentropy(label_smoothing=0.01)
mean_layer = layers.Average()


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


class ATTGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
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
        feature_real_alpha = tf.random.uniform((batch_size, self.feature_h, self.feature_w, 1),
                                               0, 1)
        feature_real_alpha = tf.cast(feature_real_alpha >= 0.75, tf.float32)
        feature_middle_alpha = tf.random.uniform((batch_size, self.feature_h, self.feature_w, 1),
                                                 0, 1)
        feature_middle_alpha = tf.cast(feature_middle_alpha >= 0.5, tf.float32)
        feature_fake_alpha = tf.random.uniform((batch_size, self.feature_h, self.feature_w, 1),
                                               0, 1)
        feature_fake_alpha = tf.cast(feature_fake_alpha >= 0.25, tf.float32)

        image_real_alpha = tf.image.resize(feature_real_alpha, (self.image_h, self.image_w),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_middle_alpha = tf.image.resize(feature_middle_alpha, (self.image_h, self.image_w),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_fake_alpha = tf.image.resize(feature_fake_alpha, (self.image_h, self.image_w),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=False) as disc_tape:
            # another domain mapping
            fake_y = self.generator([real_x, target_label])
            # Discriminator output
            blend_real_image = (real_x * image_real_alpha +
                                fake_y * (1 - image_real_alpha))
            blend_middle_image = (real_x * image_middle_alpha +
                                  fake_y * (1 - image_middle_alpha))
            blend_fake_image = (real_x * image_fake_alpha +
                                fake_y * (1 - image_fake_alpha))
            disc_real, label_real_x = self.discriminator(real_x,
                                                         training=True)
            disc_real_blend, _ = self.discriminator(blend_real_image,
                                                    training=True)
            disc_middle_blend, _ = self.discriminator(blend_middle_image,
                                                      training=True)
            disc_fake_blend, _ = self.discriminator(blend_fake_image,
                                                    training=True)
            disc_fake, _ = self.discriminator(fake_y,
                                              training=True)
            disc_real_loss = backend.mean(self.disc_real_loss_fn(disc_real))
            disc_real_blend_loss = backend.mean(self.disc_loss_fn(feature_real_alpha,
                                                                  disc_real_blend))
            disc_middle_blend_loss = backend.mean(self.disc_loss_fn(feature_middle_alpha,
                                                                    disc_middle_blend))
            disc_fake_blend_loss = backend.mean(self.disc_loss_fn(feature_fake_alpha,
                                                                  disc_fake_blend))
            disc_fake_loss = backend.mean(self.disc_fake_loss_fn(disc_fake))

            disc_loss = (disc_real_loss + disc_real_blend_loss +
                         disc_middle_blend_loss + disc_fake_blend_loss + disc_fake_loss) / 5
            label_real_x_loss = backend.mean(self.class_loss_fn(label_x,
                                                                label_real_x))

            disc_label_loss = label_real_x_loss
            disc_total_loss = disc_loss + disc_label_loss * self.disc_class_loss_coef
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
            fake_y = self.generator([real_x, target_label],
                                    training=True)
            blend_middle_image = (real_x * image_middle_alpha +
                                  fake_y * (1 - image_middle_alpha))
            recon_x = self.generator([real_x, same_label],
                                     training=True)
            # Discriminator output
            disc_fake_y, label_fake_y = self.discriminator(fake_y)
            disc_middle_blend, _ = self.discriminator(blend_middle_image,
                                                      training=True)
            label_fake_y_loss = backend.mean(self.class_loss_fn(label_y,
                                                                label_fake_y))
            gen_label_loss = label_fake_y_loss

            # Generator adverserial loss
            gen_fake_y_disc_loss = backend.mean(
                self.disc_real_loss_fn(disc_fake_y))
            gen_middle_blend_disc_loss = backend.mean(self.disc_loss_fn(1 - feature_middle_alpha,
                                                                        disc_middle_blend))

            gen_disc_loss = (gen_fake_y_disc_loss +
                             gen_middle_blend_disc_loss) / 2
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
            "disc_real_blend_loss": disc_real_blend_loss,
            "disc_middle_blend_loss": disc_middle_blend_loss,
            "disc_fake_blend_loss": disc_fake_blend_loss,
            "disc_fake_loss": disc_fake_loss,
            "disc_label_loss": disc_label_loss,
            "total_gen_loss": gen_total_loss,
            "gen_fake_y_disc_loss": gen_fake_y_disc_loss,
            "gen_middle_blend_disc_loss": gen_middle_blend_disc_loss,
            "gen_class_loss": gen_label_loss,
            "recon_x_image_loss": recon_x_image_loss,
        }
