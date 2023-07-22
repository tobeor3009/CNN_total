import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.losses import CategoricalCrossentropy

# Loss function for evaluating adversarial loss
adv_loss_fn = MeanAbsoluteError()
base_image_loss_fn = MeanAbsoluteError()
base_class_loss_fn = CategoricalCrossentropy(label_smoothing=0.01)


# Define the loss function for the generators
def base_generator_loss_deceive_discriminator(fake_img):
    return -tf.reduce_mean(fake_img)


# Define the loss function for the discriminators
def base_discriminator_loss_arrest_generator(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)

    return fake_loss - real_loss


def compute_l2_norm(tensor):

    squared = keras_backend.square(tensor)
    l2_norm = keras_backend.sum(squared)
    l2_norm = keras_backend.sqrt(l2_norm)

    return l2_norm


class ConsistencyModel(Model):
    def __init__(
        self,
        base_model
    ):
        super(ConsistencyModel, self).__init__()
        self.base_model = base_model

    def compile(
        self,
        optimizer,
        get_class_loss,
        get_consistency_seg_loss,
        get_dice_score,
        get_acc,
        class_weight=0.3,
        consistency_weight=0.7
    ):
        super(ConsistencyModel, self).compile()
        self.optimizer = optimizer
        self.get_class_loss = get_class_loss
        self.get_consistency_seg_loss = get_consistency_seg_loss
        self.get_dice_score = get_dice_score
        self.get_acc = get_acc
        self.class_weight = class_weight
        self.consistency_weight = consistency_weight

    # seems no need in usage. it just exist for keras Model's child must implement "call" method
    def call(self, x):
        return x

    def train_step(self, batch_data):
        with tf.GradientTape(persistent=True) as tape:
            loss_info, gt_info, mask_info = self.get_total_loss(batch_data)

            class_loss, consistency_seg_loss, total_loss = loss_info

        dice_score, acc = self.get_metric_results(gt_info, mask_info)
        grads = tape.gradient(total_loss,
                              self.base_model.trainable_variables)
        # Update the weights of the generators
        self.optimizer.apply_gradients(
            zip(grads, self.base_model.trainable_variables)
        )

        return {
            "class_loss": class_loss,
            "consistency_seg_loss": consistency_seg_loss,
            "total_loss": total_loss,
            "dice_score": dice_score,
            "accuracy": acc
        }

    def test_step(self, batch_data):

        loss_info, gt_info, mask_info = self.get_total_loss(batch_data)

        class_loss, consistency_seg_loss, total_loss = loss_info
        dice_score, acc = self.get_metric_results(gt_info, mask_info)
        return {
            "class_loss": class_loss,
            "consistency_seg_loss": consistency_seg_loss,
            "total_loss": total_loss,
            "dice_score": dice_score,
            "accuracy": acc
        }

    def get_total_loss(self, batch_data):
        gt_image, (gt_mask, gt_label) = batch_data
        pred_mask, pred_label = self.base_model(gt_image, training=False)
        class_loss = self.get_class_loss(gt_label, pred_label)
        consistency_seg_loss = self.get_consistency_seg_loss(gt_mask,
                                                             pred_mask, pred_label)
        total_loss = (class_loss * self.class_weight +
                      consistency_seg_loss * self.consistency_weight)
        return [(class_loss, consistency_seg_loss, total_loss),
                (gt_mask, gt_label),
                (pred_mask, pred_label)]

    def get_metric_results(self, gt_info, pred_info):
        gt_mask, gt_label = gt_info
        pred_mask, pred_label = pred_info

        dice_score = self.get_dice_score(gt_mask, pred_mask)
        acc = self.get_acc(gt_label, pred_label)

        return dice_score, acc
