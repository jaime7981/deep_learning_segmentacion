import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from loss_functions import bce_dice_loss, dice_coef
from load_dataset import load_dataset
from keras_unet_collection import models

def main():

    train_generator, test_generator = load_dataset()

    unet = models.u2net_2d(
        input_size=(128, 128, 1),
        filter_num=[64, 128, 256, 512],
        n_labels=1
    )

    unet_plus = models.u2net_plus_2d(
        input_size=(128, 128, 1),
        filter_num=[64, 128, 256, 512],
        n_labels=1
    )
    
    unet_train = unet.compile(
        optimizer=Adam,
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )

    unet_plus_train = unet_plus.compile(
        optimizer=Adam,
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )


if __name__ == '__main__':
    main()