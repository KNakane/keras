import os, sys
import keras


class LeNet():
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='tanh'))
        model.add(keras.layers.AveragePooling2D((2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh'))
        model.add(keras.layers.AveragePooling2D((2, 2), strides=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(120, activation='tanh'))
        model.add(keras.layers.Dense(84, activation='tanh'))
        model.add(keras.layers.Dense(self.out_dim, activation='softmax'))
        return model
