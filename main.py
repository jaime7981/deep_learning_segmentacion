import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from keras_unet_collection import models

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

NUM_EPOCHS = 10
BATCH_SIZE = 16

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'datasets')


def load_dataset(batch_size=32):
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    images_path = os.path.join(DATASET_PATH, 'Images')
    masks_path = os.path.join(DATASET_PATH, 'Masks')

    train_image_generator = image_datagen.flow_from_directory(
        images_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        subset='training'
    )

    train_mask_generator = mask_datagen.flow_from_directory(
        masks_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        subset='training'
    )

    test_image_generator = image_datagen.flow_from_directory(
        images_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_mask_generator = mask_datagen.flow_from_directory(
        masks_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    steps_per_epoch = len(train_image_generator)
    validation_steps = len(test_image_generator)

    train_generator = zip(train_image_generator, train_mask_generator)
    test_generator = zip(test_image_generator, test_mask_generator)

    return train_generator, test_generator, steps_per_epoch, validation_steps


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
    return celoss + (1 - dice_coef(y_true, y_pred))


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
