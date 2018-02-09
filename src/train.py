import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.losses import binary_crossentropy
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint

import argparse

def normalize_image(image):
    image = image - 128
    image = image / 128
    return(image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=False)
    args = parser.parse_args()

    input_dimensions = (100,100,3)
    imagenet_expert = Xception(include_top=False, input_shape=input_dimensions)
    for layer in imagenet_expert.layers:
        #if layer.name == 'block14_sepconv1':  # we can start with some other
        #    layer.is_layer_trainable = True
        #else:
        layer.trainable = False

    model = Sequential()
    model.add(imagenet_expert)
    model.add(Flatten())
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss = binary_crossentropy, optimizer="adam", metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint("../models/labradoodle-detector.mdl",
                                       save_best_only=True,
                                        monitor='val_acc')

    train_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator()

    model.fit_generator(train_datagen.flow_from_directory(directory=args.train,
                                                          target_size=input_dimensions[0:2],
                                                          batch_size=64,
                                                          class_mode="binary"),
                        epochs=10,
                        steps_per_epoch=10,
                        validation_data=test_datagen.flow_from_directory(directory=args.test,
                                                                         target_size=input_dimensions[0:2],
                                                                         class_mode="binary"),
                        validation_steps=10,
                        callbacks=[model_checkpoint])

if __name__ == "__main__":
    main()