import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError

from ..util.lsgan import to_real_loss, to_fake_loss
from ..util.grad_clip import adaptive_gradient_clipping

# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class Pix2PixGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        lambda_image=1,
    ):
        super(Pix2PixGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_image = lambda_image

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        image_loss=base_image_loss_fn,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1,
        lambda_disc=0.1
    ):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.image_loss = image_loss
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.lambda_clip = lambda_clip
        self.lambda_disc = lambda_disc
        self.disc_real_metric = tf.metrics.Accuracy()
        self.disc_fake_metric = tf.metrics.Accuracy()
        self.gen_fake_metric = tf.metrics.Accuracy()
        super(Pix2PixGan, self).compile()
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
        # real_x = [real_x[..., 0:1], real_x[..., 1:]]
        # disc_real_input = backend.concatenate([real_y, real_y])

        # disc_real_ratio = 0.8 - disc_real_acc
        # disc_real_ratio = tf.clip_by_value(disc_real_ratio, 0, 1)
        # disc_fake_ratio = 0.8 - disc_fake_acc
        # disc_fake_ratio = tf.clip_by_value(disc_fake_ratio, 0, 1)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as disc_tape:
            # another domain mapping
            fake_y = self.generator(real_x)
            # discriminator loss
            disc_real_y = self.discriminator(real_y, training=True)
            disc_fake_y = self.discriminator(fake_y, training=True)

            disc_real_loss = to_real_loss(disc_real_y)
            disc_fake_loss = to_fake_loss(disc_fake_y)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(disc_loss,
                                        self.discriminator.trainable_variables)
        if self.apply_adaptive_gradient_clipping:
            disc_grads = adaptive_gradient_clipping(disc_grads,
                                                    self.discriminator.trainable_variables,
                                                    lambda_clip=self.lambda_clip)
        # Update the weights of the discriminators
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )
        self.set_disc_metric(disc_real_y, disc_fake_y)
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True) as gen_tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)
            # Generator paired real y loss
            gen_loss_in_real_y = self.image_loss(real_y, fake_y)
            # Generator adverserial loss
            gen_disc_fake_y = self.discriminator(fake_y)
            gen_adverserial_loss = to_real_loss(gen_disc_fake_y)

            total_generator_loss = gen_loss_in_real_y * self.lambda_image + \
                gen_adverserial_loss * self.lambda_disc

        # Get the gradients for the generators
        gen_grads = gen_tape.gradient(total_generator_loss,
                                      self.generator.trainable_variables)
        if self.apply_adaptive_gradient_clipping is True:
            gen_grads = adaptive_gradient_clipping(
                gen_grads, self.generator.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
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
