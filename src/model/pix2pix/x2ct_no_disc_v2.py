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
        x2ct_loss=base_image_loss_fn,
        seg_loss=None,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1
    ):
        super(Pix2PixGan, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.x2ct_loss = x2ct_loss
        self.seg_loss = seg_loss
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.lambda_clip = lambda_clip
        self.disc_real_metric = tf.metrics.Accuracy()
        self.disc_fake_metric = tf.metrics.Accuracy()
        self.gen_fake_metric = tf.metrics.Accuracy()
    # seems no need in usage. it just exist for keras Model's child must implement "call" method

    def call(self, x):
        return x

    # @tf.function
    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            # another domain mapping
            fake_y = self.generator(real_x, training=True)

            x2ct_loss = self.x2ct_loss(real_y, fake_y)
            seg_loss = self.seg_loss(real_y, fake_y)
            gen_loss_in_real_y = (x2ct_loss + seg_loss) / 2
            total_generator_loss = gen_loss_in_real_y * self.lambda_image

        # Get the gradients for the generators
        gen_grads = tape.gradient(total_generator_loss,
                                  self.generator.trainable_variables)
        if self.apply_adaptive_gradient_clipping:
            gen_grads = adaptive_gradient_clipping(gen_grads,
                                                   self.generator.trainable_variables,
                                                   lambda_clip=self.lambda_clip)
        # Update the weights of the generators
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {
            "total_generator_loss": total_generator_loss,
            "x2ct_loss": x2ct_loss,
            "seg_loss": seg_loss,
            "generator_loss_in_real_y": gen_loss_in_real_y,
        }
