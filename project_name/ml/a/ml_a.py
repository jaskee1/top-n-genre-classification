
import librosa
import os
import datetime
from tensorflow import keras
import numpy as np


def get_test_and_validation_data():
    """
    Gets the test and validation data
    :return: the test and validation data
    """

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    start = datetime.datetime.now()

    # TODO: sample the data with replacement
    for folder in os.listdir('gtzan/genres/'):
        counter = 0

        for filename in os.listdir('gtzan/genres/' + folder):
            counter += 1
            file_path = 'gtzan/genres/' + folder + '/' + filename
            y, sr = librosa.load(file_path, duration=25)

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
            mel = librosa.power_to_db(mel, ref=np.max)

            if counter < 76:
                x_train.append(mel)
                y_train.append(genres.index(filename.split('.')[0]))
            else:
                x_valid.append(mel)
                y_valid.append(genres.index(filename.split('.')[0]))

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

    model = keras.models.Sequential([
        keras.layers.BatchNormalization(input_shape=(len(x_train[0]), len(x_train[0][0]), 1)),
        keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        # use softmax to normalize the probabilities
        # use 10 for 10 categories
        keras.layers.Dense(10, activation="softmax"),
    ])

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("model setup complete in " + str(elapsed_time))

    start = datetime.datetime.now()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("model compile complete" + str(elapsed_time))

    start = datetime.datetime.now()

    model.fit(x_train, y_train, epochs=5, validation_data=(x_valid, y_valid))

    end = datetime.datetime.now()
    elapsed_time = end - start

    print("model fit complete" + str(elapsed_time))


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_test_and_validation_data()
    train_model(x_train, y_train, x_valid, y_valid)

