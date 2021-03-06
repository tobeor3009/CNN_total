import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers, backend
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.losses import CategoricalCrossentropy

from .util.lsgan import to_real_loss, to_fake_loss
from .util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = CategoricalCrossentropy(label_smoothing=0.01)


def get_diff_label(label_tensor):
    label_shape = tf.shape(label_tensor)
    batch_size, num_class = label_shape[0], label_shape[1]
    label_tensor = backend.random_normal(
        shape=(batch_size, num_class)) - label_tensor
    label_tensor = tf.argmax(label_tensor, axis=1)
    label_tensor = tf.one_hot(label_tensor, depth=num_class)
    return label_tensor


class StarGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        label_mode="concatenate",
        lambda_recon=10.0,
        lambda_identity=0.5,
    ):
        super(StarGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.label_mode = label_mode
        self.lambda_recon = lambda_recon
        self.lambda_identity = lambda_identity
        self.turn_on_identity_loss = True

    def compile(
        self,
        batch_size,
        image_shape,
        generator_optimizer,
        discriminator_optimizer,
        image_loss_fn=base_image_loss_fn,
        class_loss_fn=base_class_loss_fn,
        identitiy_loss=True,
        active_gradient_clip=True,
        lambda_clip=0.1
    ):
        super(StarGan, self).compile()
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.recon_loss_fn = image_loss_fn
        self.identity_loss_fn = image_loss_fn
        self.class_loss_fn = class_loss_fn
        self.identitiy_loss = identitiy_loss,
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
        real_x, label_x = batch_data
        label_y = get_diff_label(label_x)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:

            # another domain mapping
            fake_y = self.generator([real_x, label_x, label_y])
            # back to original domain mapping
            recon_x = self.generator([fake_y, label_y, label_x])
            # identity mapping
            same_x = self.generator([real_x, label_x, label_x])
            # Discriminator output
            disc_real_x, label_pred_real_x = self.discriminator(real_x,
                                                                training=True)
            disc_fake_y, label_pred_fake_y = self.discriminator(fake_y,
                                                                training=True)
            disc_recon_x, label_pred_recon_x = self.discriminator(recon_x,
                                                                  training=True)
            disc_same_x, label_pred_same_x = self.discriminator(same_x,
                                                                training=True)
            # Compute Discriminator class_loss
            disc_real_x_class_loss = self.class_loss_fn(label_x,
                                                        label_pred_real_x)
            disc_fake_y_class_loss = self.class_loss_fn(label_y,
                                                        label_pred_fake_y)
            disc_recon_x_class_loss = self.class_loss_fn(label_x,
                                                         label_pred_recon_x)
            disc_same_x_class_loss = self.class_loss_fn(label_x,
                                                        label_pred_same_x)
            disc_class_loss = disc_real_x_class_loss + disc_fake_y_class_loss + \
                disc_recon_x_class_loss + disc_same_x_class_loss

            disc_real_x_loss = to_real_loss(disc_real_x) * 3
            disc_fake_y_loss = to_fake_loss(disc_fake_y)
            disc_recon_x_loss = to_fake_loss(disc_recon_x)
            disc_same_x_loss = to_fake_loss(disc_same_x)
            disc_loss = disc_real_x_loss + disc_fake_y_loss + \
                disc_recon_x_loss + disc_same_x_loss

            disc_total_loss = disc_class_loss + disc_loss

        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(
            disc_total_loss, self.discriminator.trainable_variables)
        if self.active_gradient_clip is True:
            disc_grads = adaptive_gradient_clipping(
                disc_grads, self.discriminator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:

            # another domain mapping
            fake_y = self.generator([real_x, label_x, label_y],
                                    training=True)
            # back to original domain mapping
            recon_x = self.generator([fake_y, label_y, label_x],
                                     training=True)
            same_x = self.generator([real_x, label_x, label_x],
                                    training=True)
            # Discriminator output
            disc_fake_y, label_pred_fake_y = self.discriminator(fake_y)
            disc_recon_x, label_pred_recon_x = self.discriminator(recon_x)
            disc_same_x, label_pred_same_x = self.discriminator(same_x)
            # Generator class loss
            gen_fake_y_class_loss = self.class_loss_fn(label_y,
                                                       label_pred_fake_y)
            gen_recon_x_class_loss = self.class_loss_fn(label_x,
                                                        label_pred_recon_x)
            gen_same_x_class_loss = self.class_loss_fn(label_x,
                                                       label_pred_same_x)
            gen_class_loss = gen_fake_y_class_loss + gen_recon_x_class_loss + \
                gen_same_x_class_loss
            # Generator adverserial loss
            gen_fake_y_disc_loss = to_real_loss(disc_fake_y)
            gen_recon_x_disc_loss = to_real_loss(disc_recon_x)
            gen_same_x_disc_loss = to_real_loss(disc_same_x)
            gen_disc_loss = gen_fake_y_disc_loss + gen_recon_x_disc_loss + \
                gen_same_x_disc_loss

            # Generator image loss
            recon_x_image_loss = self.recon_loss_fn(real_x,
                                                    recon_x) * self.lambda_recon
            same_x_image_loss = self.recon_loss_fn(
                real_x, same_x) * self.lambda_recon * self.lambda_identity
            gen_image_loss = recon_x_image_loss + same_x_image_loss
            # Total generator loss
            gen_total_loss = gen_class_loss + gen_disc_loss + gen_image_loss

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        if self.active_gradient_clip is True:
            gen_grads = adaptive_gradient_clipping(
                gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_gen_loss": gen_total_loss,
            "total_disc_loss": disc_total_loss,
            "disc_fake_y_loss": disc_fake_y_loss,
            "disc_fake_y_class_loss": disc_fake_y_class_loss,
            "gen_fake_y_disc_loss": gen_fake_y_disc_loss,
            "gen_fake_y_class_loss": gen_fake_y_class_loss,
            "gen_same_x_image_loss": same_x_image_loss,
            "gen_same_x_class_loss": gen_same_x_class_loss,
            "gen_recon_x_image_loss": recon_x_image_loss,
            "gen_recon_x_class_loss": gen_recon_x_class_loss
        }
