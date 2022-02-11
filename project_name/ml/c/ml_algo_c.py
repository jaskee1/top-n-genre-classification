from pathlib import Path

import tensorflow as tf
from tensorflow import keras

import project_name.data.feature_recorder as fr


class MlAlgoC:
    """
    """
    MODEL_PATH = Path(__file__ + '/../algo_c_model.h5')
    MODEL_RESNET_PATH = Path(__file__ + '/../algo_c_resnet_model.h5')

    def __init__(self, load_path=None,
                 input_shape=(64, 1288, 1),
                 output_shape=(10),
                 use_resnet=True):
        """
        """
        if load_path is None:
            if use_resnet:
                self.model = self._build_model_resnet_34(input_shape,
                                                         output_shape)
            else:
                self.model = self._build_model(input_shape, output_shape)
        else:
            self.load_model(load_path)

        # self.optimizer = keras.optimizers.SGD(lr=0.2,
        #  momentum=0.9, decay=0.01)

    @staticmethod
    def create_dataset(dataframe, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
        """
        """
        recorder = fr.FeatureRecorder()
        # Get bytes from tfrecord files from the input dataframe
        dataset = tf.data.TFRecordDataset(dataframe['filename'])
        # Map the raw bytes to the properly parsed data
        dataset = dataset.map(recorder.read_tfrecord_from_tfrecord_dataset,
                              num_parallel_calls=n_parse_threads)
        # Fix dimensionality -- we need an extra dimension since the CNN
        # expects an image-type input (ie, 2D, with a 3rd dimension for color
        # channels). The extra channel is essentially our version of grayscale,
        # only 1 value in the color channel.
        dataset = dataset.map(lambda x, y:
                              (tf.expand_dims(x, axis=-1), y))

        # Cache the dataset so it doesn't have to prepared again for
        # every epoch
        # dataset = dataset.cache()
        # Apply shuffling for order randomization
        dataset = dataset.shuffle(shuffle_buffer_size)
        # Get batches, with prefetching for the next batch
        return dataset.batch(batch_size).prefetch(1)

    def _build_model(self, input_shape, output_shape):
        """
        """
        model = keras.models.Sequential([
            keras.Input(shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=5, strides=2,
                                activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(128, 3, activation='relu', strides=2,
                                padding='same'),
            keras.layers.Conv2D(128, 3, activation='relu', strides=2,
                                padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(256, 3, activation='relu', strides=1,
                                padding='same'),
            keras.layers.Conv2D(256, 3, activation='relu', strides=1,
                                padding='same'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(output_shape, activation='softmax')
        ])

        print(model.summary())

        return model

    # Code source (adapted from):
    # Hands on Machine Learning with Scikit-Learn, Keras & Tensorflow
    # by Aurélien Géron
    # Chapter 14, page 478-479
    def _build_model_resnet_34(self, input_shape, output_shape):
        """
        """
        model = keras.models.Sequential()
        # Input and normalization since I'm not doing external normalization
        # of the input data
        model.add(keras.Input(shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        # Starting layer
        model.add(keras.layers.Conv2D(64, 7, strides=2,
                                      padding='same', use_bias=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2,
                                         padding='same'))

        # Repeated Residual Unit layers
        # Note that each Residual Unit has 2 main convolutional layers
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            model.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters

        # Final fully connected layers leading to output
        model.add(keras.layers.GlobalAvgPool2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))

        print(model.summary())

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


# Code source (adapted from):
# Hands on Machine Learning with Scikit-Learn, Keras & Tensorflow
# by Aurélien Géron
# Chapter 14, page 478-479
class ResidualUnit(keras.layers.Layer):
    """
    """

    def __init__(self, filters, strides=1, activation='relu', **kwargs):

        super().__init__(**kwargs)
        # Need to store these for get_config
        self.filters = filters
        self.strides = strides
        self.activation_str = activation
        # And this we actually use outside of __init__
        self._activation = keras.activations.get(activation)

        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self._activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):

        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self._activation(Z + skip_Z)

    # Needed to allow saving the model
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides,
            'activation': self.activation_str,
        })
        return config
