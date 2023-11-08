import tensorflow as tf
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_PATH, 'datasets')


def load_dataset():
    datagen_kwargs = dict (rescale = 1./255 , validation_split = 20)
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    images_path = os.path.join(DATASET_PATH, 'images')
    masks_path = os.path.join(DATASET_PATH, 'masks')

    train_image_generator = image_datagen.flow_from_directory(
        images_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )

    train_mask_generator = mask_datagen.flow_from_directory(
        masks_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )

    test_image_generator = image_datagen.flow_from_directory(
        images_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_mask_generator = mask_datagen.flow_from_directory(
        masks_path,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode=None,
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    train_generator = zip(train_image_generator, train_mask_generator)
    test_generator = zip(test_image_generator, test_mask_generator)

    return train_generator, test_generator
