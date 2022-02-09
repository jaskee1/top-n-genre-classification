# import sys
# import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# script_dir = os.path.dirname(__file__)
# mymodule_dir = os.path.join(script_dir, '..', '..')
# sys.path.append(mymodule_dir)

import project_name.data.feature_recorder as fr


class MlAlgoC:
    """
    """
    MODEL_PATH = Path(__file__ + '/../algo_c_model.h5')

    def __init__(self, load_path=None):
        """
        """
        if load_path is None:
            self.model = self._build_model()
        else:
            self.load_model(load_path)

        # self.optimizer = keras.optimizers.SGD(lr=0.2,
        #  momentum=0.9, decay=0.01)

    def create_dataset(self, dataframe, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
        """
        """
        recorder = fr.FeatureRecorder()
        # Get bytes from tfrecord files from the input dataframe
        dataset = tf.data.TFRecordDataset(dataframe['data'])
        # Map the raw bytes to the properly parsed data
        dataset = dataset.map(recorder.read_tfrecord_from_tfrecord_dataset,
                              num_parallel_calls=n_parse_threads)
        # Fix dimensionality -- we need an extra dimension since the CNN
        # expects an image-type input (ie, 2D, with a 3rd dimension for color
        # channels). The extra channel is essentially our version of grayscale,
        # only 1 value in the color channel.
        dataset = dataset.map(lambda x, y:
                              (tf.expand_dims(x, axis=-1), y))
        # Apply shuffling for order randomization
        dataset = dataset.shuffle(shuffle_buffer_size)
        # Get batches, with prefetching for the next batch
        return dataset.batch(batch_size).prefetch(1)

    def split_dataframe_gtzan(self, data, train_size=75,
                              valid_size=15, test_size=10):
        """
        """
        test_start = train_size + valid_size
        training = data[data.index % 100 < train_size]
        validation = data[
            (data.index % 100 >= train_size) & (data.index % 100 < test_start)]
        testing = data[data.index % 100 >= test_start]

        return (training, validation, testing)

    def _build_model(self):
        """
        """
        # model = keras.models.Sequential()
        # model.add(keras.layers.Flatten(input_shape=[64, 1288]))
        # model.add(keras.layers.Dense(300, activation='relu'))
        # model.add(keras.layers.Dense(100, activation='relu'))
        # model.add(keras.layers.Dense(10, activation='softmax'))

        model = keras.models.Sequential([
            keras.Input(shape=[64, 1288, 1]),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=4,
                                activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation='relu', strides=2,
                                padding='same'),
            keras.layers.Conv2D(64, 3, activation='relu', strides=2,
                                padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        # print(model.summary())

        return model

    def compile_model(self, loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy']):
        """
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def save_model(self, path=MODEL_PATH):
        """
        """
        self.model.save(path)

    def load_model(self, path=MODEL_PATH):
        """
        """
        self.model = keras.models.load_model(path)
