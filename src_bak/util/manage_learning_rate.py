class LossRatioLRManager():
    pass


def learning_rate_scheduler(
    learning_rate, epoch, schedule_list=None, exponent=0.1, warm_up=True, warm_up_epoch=20
):
    step = 0

    if epoch < 0:
        epoch = 0

    if warm_up and epoch < warm_up_epoch:
        new_learning_rate = learning_rate * ((epoch + 1) / warm_up_epoch)
    else:
        if schedule_list is None:
            schedule_list = [40, 60, 80, 100, 120, 140, 160, 200]
        for step, target_epoch in enumerate(schedule_list):
            if target_epoch >= epoch:
                break
            else:
                continue
        new_learning_rate = learning_rate * (exponent ** (step))

    return new_learning_rate


def poly_learning_rate_scheduler(learning_rate, epoch, max_epoch=300, exponent=0.9):
    # poly_lr
    new_learning_rate = learning_rate * (1 - epoch / max_epoch) ** exponent

    return new_learning_rate
