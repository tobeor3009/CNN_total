import torch

def propotional_dice_loss(y_true, y_pred, beta=0.7, smooth=1e-7, channel_weight=None):

    alpha = 1 - beta
    prevalence = y_true.mean(dim=(2,3))

    tp = (y_true * y_pred).sum(dim=(2,3))
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=(2,3))
    fp = (y_pred).sum(dim=(2,3)) - tp
    fn = (y_true).sum(dim=(2,3)) - tp

    negative_score = (tn + smooth) \
        / (tn + beta * fn + alpha * fp + smooth) * (smooth + 1 - prevalence)
    positive_score = (tp + smooth) \
        / (tp + alpha * fn + beta * fp + smooth) * (smooth + prevalence)

    total_score = (negative_score + positive_score)
    total_score = -1 * torch.log(total_score)
    if channel_weight is not None:
        channel_weight_shape = (-1, 1, 1, channel_weight.shape[0])
        channel_weight = channel_weight.reshape(channel_weight_shape)
        total_score = total_score * channel_weight

    return total_score.mean()