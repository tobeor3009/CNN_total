from tensorflow.keras import backend


def to_real_loss(y_pred):
    y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
    term_1 = backend.log(y_pred + backend.epsilon())
    return -backend.mean(term_1)


def to_fake_loss(y_pred):
    y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
    term_0 = backend.log(1 - y_pred + backend.epsilon())
    return -backend.mean(term_0)


def BinaryCrossEntropy(y_true, y_pred):
    y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
    term_0 = (1 - y_true) * backend.log(1 - y_pred + backend.epsilon())
    term_1 = y_true * backend.log(y_pred + backend.epsilon())
    return -backend.mean(term_0 + term_1)
