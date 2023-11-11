import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from loss_functions import bce_dice_loss, dice_coef
from load_dataset import load_dataset
from keras_unet_collection import models

NUM_EPOCHS = 32
BATCH_SIZE = 10

def main():
    train_generator, test_generator, steps_per_epoch, validation_steps = load_dataset(batch_size=BATCH_SIZE)

    # U-Net Model
    unet = models.unet_2d(
        input_size=(128, 128, 1),
        filter_num=[64, 128, 256, 512],
        n_labels=1
    )

    unet.compile(
        optimizer=Adam(),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )

    # Train U-Net Model
    unet.fit(
        train_generator,
        epochs=NUM_EPOCHS,
        validation_data=test_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    unet.save('unet.h5')

    # U-Net++ Model
    unet_plus = models.unet_plus_2d(
        input_size=(128, 128, 1),
        filter_num=[64, 128, 256, 512],
        n_labels=1
    )
    
    unet_plus.compile(
        optimizer=Adam(),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )

    # Train U-Net++ Model
    unet_plus.fit(
        train_generator,
        epochs=NUM_EPOCHS,
        validation_data=test_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    unet_plus.save('unet_plus.h5')

if __name__ == '__main__':
    main()
