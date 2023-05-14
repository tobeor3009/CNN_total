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


def base_ssim_loss_fn(y_true, y_pred):
    ssim_score = tf.image.ssim(y_true, y_pred, max_val=1)
    ssim_loss = 1 - ssim_score
    return ssim_loss


mean_layer = layers.Average()


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

    def compile(
        self,
        batch_size,
        image_shape,
        generator_optimizer,
        discriminator_optimizer,
        image_loss_fn=base_image_loss_fn,
        class_loss_fn=base_class_loss_fn,
        disc_real_loss_fn=to_real_loss,
        disc_fake_loss_fn=to_fake_loss,
        active_gradient_clip=True,
        image_loss_coef=100.0,
        gen_class_loss_coef=10.0,
        disc_class_loss_coef=1.0,
        gen_disc_loss_coef=1.0,
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
        self.gen_class_loss_coef = gen_class_loss_coef
        self.disc_class_loss_coef = disc_class_loss_coef
        self.disc_real_loss_fn = disc_real_loss_fn
        self.disc_fake_loss_fn = disc_fake_loss_fn
        self.gen_disc_loss_coef = gen_disc_loss_coef

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
        label_y = get_diff_label(label_x)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator([real_x, label_y])
            # Discriminator output
            disc_real_x, label_pred_real_x = self.discriminator(real_x,
                                                                training=True)
            disc_fake_y, _ = self.discriminator(fake_y,
                                                training=True)
            # Compute Discriminator class_loss
            disc_real_x_class_loss = backend.mean(self.class_loss_fn(label_x,
                                                                     label_pred_real_x))
            disc_class_loss = disc_real_x_class_loss

            disc_real_x_loss = self.disc_real_loss_fn(disc_real_x)
            disc_fake_y_loss = self.disc_fake_loss_fn(disc_fake_y)
            disc_loss = (disc_real_x_loss + disc_fake_y_loss) / 2

            disc_total_loss = disc_class_loss * self.disc_class_loss_coef + disc_loss

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
        with tf.GradientTape(persistent=True) as gen_tape:

            # another domain mapping
            fake_y = self.generator([real_x, label_y],
                                    training=True)
            same_x = self.generator([real_x, label_x],
                                    training=True)
            # Discriminator output
            disc_fake_y, label_pred_fake_y = self.discriminator(fake_y)
            # Generator class loss
            gen_fake_y_class_loss = backend.mean(self.class_loss_fn(label_y,
                                                                    label_pred_fake_y))
            gen_class_loss = gen_fake_y_class_loss
            # Generator adverserial loss
            gen_fake_y_disc_loss = self.disc_real_loss_fn(disc_fake_y)
            gen_disc_loss = gen_fake_y_disc_loss

            # Generator image loss
            same_x_image_loss = self.recon_loss_fn(real_x,
                                                   same_x)
            gen_image_loss = backend.mean(same_x_image_loss)
            # Total generator loss
            gen_total_loss = gen_class_loss * self.gen_class_loss_coef +\
                gen_disc_loss * self.gen_disc_loss_coef + gen_image_loss * self.image_loss_coef

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(gen_total_loss,
                                      self.generator.trainable_variables)
        if self.active_gradient_clip is True:
            gen_grads = adaptive_gradient_clipping(gen_grads,
                                                   self.generator.trainable_variables,
                                                   lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_gen_loss": gen_total_loss,
            "total_disc_loss": disc_total_loss,
            "disc_real_x_loss": disc_real_x_loss,
            "disc_fake_y_loss": disc_fake_y_loss,
            "disc_class_loss": disc_class_loss,
            "gen_fake_y_disc_loss": gen_fake_y_disc_loss,
            "gen_fake_y_class_loss": gen_fake_y_class_loss,
            "gen_same_x_image_loss": same_x_image_loss,
        }
