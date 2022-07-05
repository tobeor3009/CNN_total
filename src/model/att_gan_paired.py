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


def split_multiclass_into_batch(real_img, label, num_class):
    img_shape = tf.shape(real_img)
    b, h, w, c = img_shape[0], img_shape[1], img_shape[2], img_shape[3]
    target_label = backend.random_normal(shape=(b, num_class))
    target_label = tf.argmax(target_label, axis=1)
    target_label = tf.one_hot(target_label, depth=num_class)
    target_label_idx = tf.argmax(target_label, axis=1, output_type="int32")
    target_label = backend.expand_dims(target_label, axis=1)
    target_label = backend.repeat_elements(target_label, rep=num_class, axis=1)

    target_img = []
    for idx in range(len(target_label_idx)):
        real_idx = target_label_idx[idx]
        real_img_partial = tf.gather(real_img, indices=idx,
                                     axis=0)
        real_img_partial = tf.gather(real_img_partial, indices=real_idx,
                                     axis=-1)
        target_img.append(real_img_partial)
    target_img = backend.stack(target_img, axis=0)
    target_img = backend.repeat_elements(target_img, rep=num_class, axis=0)

    real_img = backend.permute_dimensions(real_img, (0, 4, 1, 2, 3))
    real_img = backend.reshape(real_img, (b * num_class, h, w, c))

    label = backend.reshape(label, (b * num_class, num_class))
    target_label = backend.reshape(target_label, (b * num_class, num_class))

    return real_img, target_img, label, target_label


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
        num_class,
        generator_optimizer,
        discriminator_optimizer,
        image_loss_fn=base_image_loss_fn,
        class_loss_fn=base_class_loss_fn,
        active_gradient_clip=True,
        image_loss_coef=100.0,
        gen_class_loss_coef=10.0,
        disc_class_loss_coef=1.0,
        gen_dics_loss_coef=1.0,
        lambda_clip=0.1
    ):
        super(ATTGan, self).compile()
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_class = num_class
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.recon_loss_fn = image_loss_fn
        self.class_loss_fn = class_loss_fn
        self.active_gradient_clip = active_gradient_clip
        self.lambda_clip = lambda_clip
        self.image_loss_coef = image_loss_coef
        self.gen_class_loss_coef = gen_class_loss_coef
        self.disc_class_loss_coef = disc_class_loss_coef
        self.gen_dics_loss_coef = gen_dics_loss_coef

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

    @tf.function
    def train_step(self, batch_data):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        (real_img, target_img), (label, target_label) = batch_data
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_img = self.generator([real_img, target_label])
            # Discriminator output
            disc_real, label_pred_real = self.discriminator(real_img,
                                                            training=True)
            disc_fake, _ = self.discriminator(fake_img,
                                              training=True)
            # Compute Discriminator class_loss
            disc_real_class_loss = self.class_loss_fn(label,
                                                      label_pred_real)
            disc_class_loss = disc_real_class_loss

            disc_real_loss = to_real_loss(disc_real)
            disc_fake_loss = to_fake_loss(disc_fake)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

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
            fake_img = self.generator([real_img, target_label],
                                      training=True)
            # Discriminator output
            disc_fake, label_pred_fake = self.discriminator(fake_img)
            # Generator class loss
            gen_fake_class_loss = self.class_loss_fn(target_label,
                                                     label_pred_fake)
            gen_class_loss = gen_fake_class_loss
            # Generator adverserial loss
            gen_fake_disc_loss = to_real_loss(disc_fake)
            gen_disc_loss = gen_fake_disc_loss

            # Generator image loss
            fake_image_loss = self.recon_loss_fn(target_img,
                                                 fake_img)
            gen_image_loss = fake_image_loss
            # Total generator loss
            gen_total_loss = gen_class_loss * self.gen_class_loss_coef + \
                gen_disc_loss * self.gen_dics_loss_coef + gen_image_loss * self.image_loss_coef

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
            "disc_real_loss": disc_real_loss,
            "disc_fake_loss": disc_fake_loss,
            "gen_fake_image_loss": fake_image_loss,
            "gen_fake_disc_loss": gen_fake_disc_loss,
            "gen_fake_class_loss": gen_fake_class_loss
        }
