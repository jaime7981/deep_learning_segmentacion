import tensorflow as tf
from loss_functions import bce_dice_loss, dice_coef
from tensorflow.keras.optimizers import Adam

def main():
    
    model = model.compile(
        optimizer=Adam,
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )


if __name__ == '__main__':
    main()