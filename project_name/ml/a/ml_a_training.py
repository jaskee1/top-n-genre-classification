"""
This is a stand alone script that trains and saves the model A
"""

import librosa
import os
import datetime
from tensorflow import keras
import numpy as np
import random
from keras import callbacks

TRAINING_FILE_PATH = 'fma_small_split_wav/fma_train/'
VALIDATION_FILE_PATH = 'fma_small_split_wav/fma_validate/'

def get_test_and_validation_data():
    """
    Gets the test and validation data
    :return: the test and validation data
    """
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    genres = ["electronic", "experimental", "folk", "hiphop", "instrumental", "international", "pop", "rock"]

    start = datetime.datetime.now()
    counter = 0

    while counter < 6397:
        # citation - solution for choosing random file from
        # https://stackoverflow.com/questions/701402/best-way-to-choose-a-random-file-from-a-directory
        folder = random.choice(os.listdir(TRAINING_FILE_PATH))
        filename = random.choice(os.listdir(TRAINING_FILE_PATH + folder))

        print(folder + "/" + filename)
        try:
            file_path = TRAINING_FILE_PATH + folder + '/' + filename
            y, sr = librosa.load(file_path, offset=5, duration=20)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, n_mfcc=100)

            x_train.append(mfcc)
            y_train.append(genres.index(folder))

            counter += 1
            print(counter)
        except Exception as e:
            print("exception on " + filename)

    for folder in os.listdir(VALIDATION_FILE_PATH):
        for filename in os.listdir(VALIDATION_FILE_PATH + folder):
            try:
                file_path = VALIDATION_FILE_PATH + folder + '/' + filename
                y, sr = librosa.load(file_path, offset=5, duration=20)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, n_mfcc=100)

                x_valid.append(mfcc)
                y_valid.append(genres.index(folder))
            except Exception as e:
                print("exception on " + filename)

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("data parsing and preparation complete in " + str(elapsed_time))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_valid = np.asarray(x_valid)
    y_valid = np.asarray(y_valid)

    return x_train, y_train, x_valid, y_valid


def train_model(x_train, y_train, x_valid, y_valid):
    """
    trains the CNN
    :return:
    """

    # add fourth dimension
    x_train = x_train.reshape(len(x_train), len(x_train[0]), len(x_train[0][0]), 1)
    x_valid = x_valid.reshape(len(x_valid), len(x_valid[0]), len(x_valid[0][0]), 1)

    start = datetime.datetime.now()

    # citation - closely based on example from
    # Hands-On Machine Learning Using SciKit, Keras and Tensorflow pg 461
    model = keras.models.Sequential([
        keras.layers.BatchNormalization(input_shape=(len(x_train[0]), len(x_train[0][0]), 1)),
        keras.layers.Conv2D(64, 7, activation="relu", padding="same"),  # CNN layers
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),  # ANN layers
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        # use softmax to normalize the probabilities
        # use 8 for 8 categories
        keras.layers.Dense(8, activation="softmax"),
    ])

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("model setup complete in " + str(elapsed_time))

    start = datetime.datetime.now()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"],)

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("model compile complete " + str(elapsed_time))

    start = datetime.datetime.now()

    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min", patience=10,
                                            restore_best_weights=True)

    model.fit(x_train, y_train, epochs=200, steps_per_epoch=40, validation_data=(x_valid, y_valid), callbacks=[earlystopping])

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("model fit complete " + str(elapsed_time))

    model.save("model_a_50.h5")


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_test_and_validation_data()
    train_model(x_train, y_train, x_valid, y_valid)

