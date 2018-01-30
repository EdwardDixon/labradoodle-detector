import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import Xception
import argparse



def main():
    # TODO commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=False)
    args = parser.parse_args()

    input_dimensions = (128,128,3)
    imagenet_expert = Xception(include_top=False, input_shape=input_dimensions)
    for layer in imagenet_expert.layers:
        layer.trainable = False

    model = Sequential()
    model.add(imagenet_expert)
    model.add(Flatten())
    model.add(Dense(2, activation="relu"))

    model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    datagen = ImageDataGenerator()
    model.fit_generator(datagen.flow_from_directory(args.train, target_size=input_dimensions[0:2]),
                        steps_per_epoch=10,
                        validation_data=datagen.flow_from_directory(directory=args.test, target_size=input_dimensions[0:2]),
                        validation_steps=10)

if __name__ == "__main__":
    main()