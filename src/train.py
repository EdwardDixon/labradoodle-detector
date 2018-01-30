import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import Xception
import argparse



def main():
    # TODO commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()

    input_dimensions = (128,128)
    imagenet_expert = Xception(include_top=False, input_shape=input_dimensions)
    for layer in imagenet_expert:
        layer.trainable = False

    model = Sequential()
    model.add(imagenet_expert)
    model.add(Flatten())
    model.add(Dense(2, activation="relu"))

    model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    datagen = ImageDataGenerator()
    datagen.flow_from_directory(directory=args.data)
    model.fit_generator(ImageDataGenerator(datagen.flow_from_directory(args.data)))

if __name__ == "__main__":
    main()