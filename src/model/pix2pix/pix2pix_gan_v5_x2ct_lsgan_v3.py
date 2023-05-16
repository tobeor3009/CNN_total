import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.losses import MeanAbsoluteError

from ..util.binary_crossentropy import to_real_loss, to_fake_loss
from ..util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class Pix2PixGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        to_real_loss=to_real_loss,
        to_fake_loss=to_fake_loss,
        lambda_image=10.0,
        lambda_disc=0.1
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.to_real_loss = to_real_loss
        self.to_fake_loss = to_fake_loss
        self.lambda_image = lambda_image
        self.lambda_disc = lambda_disc

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss=base_image_loss_fn,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.image_loss = image_loss
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.lambda_clip = lambda_clip
        self.disc_real_metric = tf.metrics.Accuracy()
        self.disc_fake_metric = tf.metrics.Accuracy()
        self.gen_fake_metric = tf.metrics.Accuracy()
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    # def build(self, input_shape):
    #     self.disc_real_metric.update_state([[1], [1]], [[1], [1]])
    #     self.disc_fake_metric.update_state([[0], [0]], [[0], [0]])
    #     self.gen_fake_metric.update_state([[1], [1]], [[1], [1]])

    def get_metric_result(self, metric):
        result = metric.result()
        metric.reset_states()
        return result

    def set_gen_metric(self, fake_data):
        self.gen_fake_metric.update_state(tf.ones_like(fake_data) >= 0.5,
                                          fake_data >= 0.5)

    def set_disc_metric(self, real_data, fake_data):
        self.disc_real_metric.update_state(tf.ones_like(real_data) >= 0.5,
                                           real_data >= 0.5)
        self.disc_fake_metric.update_state(tf.zeros_like(fake_data) < 0.5,
                                           fake_data < 0.5)

    # @tf.function
    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, real_y = batch_data
        real_ap_y = backend.mean(real_y, axis=2)
        real_lat_y = backend.mean(real_y, axis=3)
        real_disc_y = backend.concatenate([real_ap_y,
                                           real_lat_y], axis=-1)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            fake_ap_y = backend.mean(fake_y, axis=2)
            fake_lat_y = backend.mean(fake_y, axis=3)
            fake_disc_y = backend.concatenate([fake_ap_y,
                                               fake_lat_y], axis=-1)
            # discriminator loss
            disc_real_y = self.discriminator(real_disc_y, training=True)
            disc_fake_y = self.discriminator(fake_disc_y, training=True)

            disc_real_loss = self.to_real_loss(disc_real_y)
            disc_fake_loss = self.to_fake_loss(disc_fake_y)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            gen_disc_fake_y = self.discriminator(fake_disc_y, training=False)
            gen_adverserial_loss = self.to_real_loss(gen_disc_fake_y)
            total_generator_loss = gen_loss_in_real_y * self.lambda_image + \
                gen_adverserial_loss * self.lambda_disc

        # Get the gradients for the discriminators
        disc_grads = tape.gradient(disc_loss,
                                   self.discriminator.trainable_variables)
        # Get the gradients for the generators
        gen_grads = tape.gradient(total_generator_loss,
                                  self.generator.trainable_variables)
        if self.apply_adaptive_gradient_clipping:
            disc_grads = adaptive_gradient_clipping(disc_grads,
                                                    self.discriminator.trainable_variables,
                                                    lambda_clip=self.lambda_clip)
            gen_grads = adaptive_gradient_clipping(gen_grads,
                                                   self.generator.trainable_variables,
                                                   lambda_clip=self.lambda_clip)
        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        self.set_disc_metric(disc_real_y, disc_fake_y)
        self.set_gen_metric(gen_disc_fake_y)

        disc_real_acc = self.get_metric_result(self.disc_real_metric)
        disc_fake_acc = self.get_metric_result(self.disc_fake_metric)
        gen_fake_acc = self.get_metric_result(self.gen_fake_metric)

        return {
            "total_generator_loss": total_generator_loss,
            "generator_loss_in_real_y": gen_loss_in_real_y,
            "gen_disc_loss": gen_adverserial_loss,
            "disc_real_loss": disc_real_loss,
            "disc_fake_loss": disc_fake_loss,
            "disc_real_acc": disc_real_acc,
            "disc_fake_acc": disc_fake_acc,
            "gen_fake_acc": gen_fake_acc
        }
