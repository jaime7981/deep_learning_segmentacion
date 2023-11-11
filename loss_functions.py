import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    up = (2. * intersection + smooth)
    down = (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice = up / down
    return dice


def bce_dice_loss(y_true, y_pred):
    celoss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return celoss + dice_coef(y_true, y_pred)