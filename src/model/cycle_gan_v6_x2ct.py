import tensorflow as tf
from tensorflow.keras import backend as keras_backend
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras import layers, backend
from tensorflow.keras import activations
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal

from .util.lsgan import to_real_loss
from .util.lsgan import to_fake_loss
from .util.grad_clip import adaptive_gradient_clipping

# Define Base Model Params
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
gamma_init = RandomNormal(mean=0.0, stddev=0.02)
CHECK_SLICE_NUM = 8
# Loss function for evaluating adversarial loss
base_image_loss_fn = MeanAbsoluteError()


class CycleGan(Model):
    def __init__(
        self,
        gen_2d_3d,
        gen_3d_2d,
        disc_2d,
        ct_size,
        n_drr,
        to_real_loss=to_real_loss,
        to_fake_loss=to_fake_loss,
        lambda_image_loss_2d=10.0,
        lambda_image_loss_3d=10.0,
        lambda_gen_2_disc=1.0,
        gp_weight=10.0

    ):
        super(CycleGan, self).__init__()
        self.gen_2d_3d = gen_2d_3d
        self.gen_3d_2d = gen_3d_2d
        self.disc_2d = disc_2d
        self.to_real_loss = to_real_loss
        self.to_fake_loss = to_fake_loss
        self.lambda_gen_2_disc = lambda_gen_2_disc
        self.lambda_image_loss_2d = lambda_image_loss_2d
        self.lambda_image_loss_3d = lambda_image_loss_3d
        self.turn_on_discriminator_on_identity = False
        self.gp_weight = gp_weight
        self.ct_size = ct_size
        self.n_drr = n_drr
        self.slice_shape = (-1, self.ct_size, self.ct_size, 1)

    def compile(
        self,
        gen_2d_3d_optim,
        gen_3d_2d_optim,
        disc_2d_optim,
        image_2d_loss_fn=base_image_loss_fn,
        image_3d_loss_fn=base_image_loss_fn,
        apply_adaptive_gradient_clipping=True,
        lambda_clip=0.1,
    ):
        super(CycleGan, self).compile()
        self.gen_2d_3d_optim = gen_2d_3d_optim
        self.gen_3d_2d_optim = gen_3d_2d_optim
        self.disc_2d_optim = disc_2d_optim
        self.image_2d_loss_fn = image_2d_loss_fn
        self.image_3d_loss_fn = image_3d_loss_fn
        self.apply_adaptive_gradient_clipping = apply_adaptive_gradient_clipping
        self.lambda_clip = lambda_clip

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    def get_random_slice_2d(self, data_2d):
        slice_data_2d = backend.permute_dimensions(data_2d,
                                                   (0, 3, 1, 2))
        slice_data_2d = backend.reshape(slice_data_2d, self.slice_shape)
        return slice_data_2d

    def get_random_slice_3d(self, data_3d, slice_rand_idx, slice_num):

        slice_data_3d = data_3d[:, slice_rand_idx:slice_rand_idx + slice_num]
        slice_data_3d = backend.permute_dimensions(slice_data_3d,
                                                   (0, 4, 1, 2, 3))
        slice_data_3d = backend.reshape(slice_data_3d, self.slice_shape)
        return slice_data_3d

    def disc_2d_to_batch_size(self, disc_2d, n_drr):
        batch_disc_2d = backend.reshape(disc_2d, (-1, n_drr))
        batch_disc_2d = backend.mean(batch_disc_2d, axis=1)
        return batch_disc_2d

    def disc_3d_to_batch_size(self, disc_3d, slice_num):
        batch_disc_3d = backend.reshape(disc_3d, (-1, slice_num))
        batch_disc_3d = backend.mean(batch_disc_3d, axis=1)
        return batch_disc_3d

        # @tf.function
    def train_step(self, batch_data):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        real_2d, real_3d = batch_data
        slice_rand_idx = tf.random.uniform(shape=(), minval=0, maxval=self.ct_size - CHECK_SLICE_NUM,
                                           dtype=tf.int32)
        slice_real_2d = self.get_random_slice_2d(real_2d)
        slice_real_3d = self.get_random_slice_3d(real_3d,
                                                 slice_rand_idx, CHECK_SLICE_NUM)
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as disc_tape:
            # another domain mapping
            fake_2d = self.gen_3d_2d(real_3d)
            slice_fake_2d = self.get_random_slice_2d(fake_2d)
            fake_3d = self.gen_2d_3d(real_2d)
            slice_fake_3d = self.get_random_slice_3d(fake_3d,
                                                     slice_rand_idx, CHECK_SLICE_NUM)
            # back to original domain mapping
            cycle_2d = self.gen_3d_2d(fake_3d)
            slice_cycle_2d = self.get_random_slice_2d(cycle_2d)
            cycle_3d = self.gen_2d_3d(fake_2d)
            slice_cycle_3d = self.get_random_slice_3d(cycle_3d,
                                                      slice_rand_idx, CHECK_SLICE_NUM)

            # Discriminator output
            disc_real_2d = self.disc_2d(slice_real_2d, training=True)
            disc_fake_2d = self.disc_2d(slice_fake_2d, training=True)
            disc_cycle_2d = self.disc_2d(slice_cycle_2d, training=True)

            disc_slice_real_3d = self.disc_2d(slice_real_3d, training=True)
            disc_slice_fake_3d = self.disc_2d(slice_fake_3d, training=True)
            disc_slice_cycle_3d = self.disc_2d(slice_cycle_3d, training=True)

            # Discriminator loss
            disc_2d_fake_loss = (self.to_real_loss(disc_real_2d) +
                                 self.to_fake_loss(disc_fake_2d))
            disc_3d_fake_loss = (self.to_real_loss(disc_slice_real_3d) +
                                 self.to_fake_loss(disc_slice_fake_3d))
            disc_2d_cycle_loss = self.to_fake_loss(disc_cycle_2d)
            disc_3d_cycle_loss = self.to_fake_loss(disc_slice_cycle_3d)

            disc_2d_fake_loss = self.disc_2d_to_batch_size(
                disc_2d_fake_loss, self.n_drr)
            disc_2d_cycle_loss = self.disc_2d_to_batch_size(
                disc_2d_cycle_loss, self.n_drr)
            disc_3d_fake_loss = self.disc_3d_to_batch_size(
                disc_3d_fake_loss, CHECK_SLICE_NUM)
            disc_3d_cycle_loss = self.disc_3d_to_batch_size(
                disc_3d_cycle_loss, CHECK_SLICE_NUM)

            disc_2d_total_loss = (disc_2d_fake_loss + disc_2d_cycle_loss)
            disc_3d_total_loss = (disc_3d_fake_loss + disc_3d_cycle_loss)
            disc_total_loss = disc_2d_total_loss + disc_3d_total_loss
        # Get the gradients for the discriminators
        disc_grads = disc_tape.gradient(disc_total_loss,
                                        self.disc_2d.trainable_variables)

        if self.apply_adaptive_gradient_clipping is True:
            # Apply Active Gradient Clipping on discriminator's grad
            disc_grads = adaptive_gradient_clipping(
                disc_grads, self.disc_2d.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the discriminators
        self.disc_2d_optim.apply_gradients(
            zip(disc_grads, self.disc_2d.trainable_variables)
        )

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as gen_tape:

            # another domain mapping
            fake_2d = self.gen_3d_2d(real_3d, training=True)
            slice_fake_2d = self.get_random_slice_2d(fake_2d)
            fake_3d = self.gen_2d_3d(real_2d, training=True)
            slice_fake_3d = self.get_random_slice_3d(fake_3d,
                                                     slice_rand_idx, CHECK_SLICE_NUM)
            # back to original domain mapping
            cycle_2d = self.gen_3d_2d(fake_3d, training=True)
            slice_cycle_2d = self.get_random_slice_2d(cycle_2d)
            cycle_3d = self.gen_2d_3d(fake_2d, training=True)
            slice_cycle_3d = self.get_random_slice_3d(cycle_3d,
                                                      slice_rand_idx, CHECK_SLICE_NUM)

            # Discriminator output
            disc_fake_2d = self.disc_2d(slice_fake_2d)
            disc_slice_fake_3d = self.disc_2d(slice_fake_3d)

            disc_cycle_2d = self.disc_2d(slice_cycle_2d)
            disc_slice_cycle_3d = self.disc_2d(slice_cycle_3d)

            # Generator image loss
            gen_2d_3d_fake_image_loss = self.image_3d_loss_fn(
                real_3d, fake_3d) * self.lambda_image_loss_3d
            gen_3d_2d_fake_image_loss = self.image_2d_loss_fn(
                real_2d, fake_2d) * self.lambda_image_loss_2d
            gen_2d_3d_cycle_image_loss = self.image_3d_loss_fn(
                real_3d, cycle_3d) * self.lambda_image_loss_3d
            gen_3d_2d_cycle_image_loss = self.image_2d_loss_fn(
                real_2d, cycle_2d) * self.lambda_image_loss_2d

            gen_2d_3d_fake_disc_loss = self.to_real_loss(disc_slice_fake_3d)
            gen_3d_2d_fake_disc_loss = self.to_real_loss(disc_fake_2d)

            gen_2d_3d_cycle_disc_loss = self.to_real_loss(disc_slice_cycle_3d)
            gen_3d_2d_cycle_disc_loss = self.to_real_loss(disc_cycle_2d)

            gen_3d_2d_fake_disc_loss = self.disc_2d_to_batch_size(gen_3d_2d_fake_disc_loss,
                                                                  self.n_drr)
            gen_3d_2d_cycle_disc_loss = self.disc_2d_to_batch_size(gen_3d_2d_cycle_disc_loss,
                                                                   self.n_drr)
            gen_2d_3d_fake_disc_loss = self.disc_3d_to_batch_size(gen_2d_3d_fake_disc_loss,
                                                                  CHECK_SLICE_NUM)
            gen_2d_3d_cycle_disc_loss = self.disc_3d_to_batch_size(gen_2d_3d_cycle_disc_loss,
                                                                   CHECK_SLICE_NUM)

            gen_2d_3d_disc_loss = (gen_2d_3d_fake_disc_loss +
                                   gen_2d_3d_cycle_disc_loss) * self.lambda_gen_2_disc
            gen_3d_2d_disc_loss = (gen_3d_2d_fake_disc_loss +
                                   gen_3d_2d_cycle_disc_loss) * self.lambda_gen_2_disc
            gen_2d_3d_image_loss = (gen_2d_3d_fake_image_loss +
                                    gen_2d_3d_cycle_image_loss)
            gen_3d_2d_fake_image_loss = backend.mean(gen_3d_2d_fake_image_loss,
                                                     axis=[1, 2])
            gen_3d_2d_cycle_image_loss = backend.mean(gen_3d_2d_cycle_image_loss,
                                                      axis=[1, 2])
            gen_3d_2d_image_loss = (gen_3d_2d_fake_image_loss +
                                    gen_3d_2d_cycle_image_loss)

            # Total generator loss
            gen_2d_3d_total_loss = (gen_2d_3d_image_loss + gen_2d_3d_disc_loss)
            gen_3d_2d_total_loss = (gen_3d_2d_image_loss + gen_3d_2d_disc_loss)

        # Get the gradients for the generators
        gen_2d_3d_grads = gen_tape.gradient(gen_2d_3d_total_loss,
                                            self.gen_2d_3d.trainable_variables)
        gen_3d_2d_grads = gen_tape.gradient(gen_3d_2d_total_loss,
                                            self.gen_3d_2d.trainable_variables)

        if self.apply_adaptive_gradient_clipping is True:
            # Apply Active Gradient Clipping on generator's grad
            gen_2d_3d_grads = adaptive_gradient_clipping(
                gen_2d_3d_grads, self.gen_2d_3d.trainable_variables, lambda_clip=self.lambda_clip)
            gen_3d_2d_grads = adaptive_gradient_clipping(
                gen_3d_2d_grads, self.gen_3d_2d.trainable_variables, lambda_clip=self.lambda_clip)

        # Update the weights of the generators
        self.gen_2d_3d_optim.apply_gradients(
            zip(gen_2d_3d_grads, self.gen_2d_3d.trainable_variables)
        )
        self.gen_3d_2d_optim.apply_gradients(
            zip(gen_3d_2d_grads, self.gen_3d_2d.trainable_variables)
        )
        return {
            "total_loss_G": gen_2d_3d_total_loss,
            "total_loss_F": gen_3d_2d_total_loss,
            "disc_2d_loss": disc_2d_fake_loss,
            "disc_3d_loss": disc_3d_fake_loss,
            "gen_2d_3d_loss": gen_2d_3d_fake_disc_loss,
            "gen_3d_2d_loss": gen_3d_2d_fake_disc_loss,
            "fake_image_loss_2d": gen_3d_2d_fake_image_loss,
            "fake_image_loss_3d": gen_2d_3d_fake_image_loss,
            "cycle_image_loss_2d": gen_3d_2d_cycle_image_loss,
            "cycle_image_loss_3d": gen_2d_3d_cycle_image_loss
        }
