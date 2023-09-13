import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.losses import CategoricalCrossentropy


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

        dice_score = self.get_dice_score(gt_mask,
                                         pred_mask * pred_label[:, None, None, :])
        acc = self.get_acc(gt_label, pred_label)

        return dice_score, acc
