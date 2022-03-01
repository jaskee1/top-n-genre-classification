# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import keras
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import (Dense, BatchNormalization, Flatten,
                          Conv2D, Dropout, MaxPooling2D)
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



class MlAlgoB:
    _accepted_data_sources = ("gtzan", "fma")

    def __init__(self, data_source: str = "gtzan", classes: int = 10,
                 cwd: str = os.getcwd()):

        # Only accepting data sources that are currently implemented
        if data_source.lower() not in self._accepted_data_sources:
            raise ValueError(
                f'{data_source} is not currently implemented. Options are: \
                  {self._accepted_data_sources}')
        else:
            self._data_source = data_source

        # Setting current working directory
        self.cwd = cwd

        # Directory of model class
        self._model_path = os.getcwd() + '\\algo_b_model.h5'

        # Loading existing CNN model if exists, else creating new one
        if os.path.exists(self._model_path):
            self.load_model(self._model_path)
            print("Model loaded")
        else:
            self.model = self.create_genre_cnn_model(classes = classes)
            print("No model found")

    def create_genre_cnn_model(self, input_shape: tuple = (288, 432, 4),
                               classes: int = 10):
        """ Creates a cnn model from the given inputs

        Keyword arguments:
        input_shape -- Shape of the input file (height, width, channels)
        classes -- Number of categorical labels
        """

        model = keras.Sequential()

        # 1st Conv2D layer
        model.add(Conv2D(8, (3, 3), strides=(1, 1),
                  activation='relu', input_shape=input_shape))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2nd Conv2D layer
        model.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3rd Conv2D layer
        model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 4th Conv2D layer
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 5th Conv2D layer
        model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding dropout and dense top layers, then returning
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model

    def create_data_generators(self, file_prefix = None, batch_size: int = 8):
        """ Creates data generators from mel .pngs to feed model training

        Keyword arguments:
        batch_size -- Number of samples to examine before modifying hyper 
        parameters
        """
        
        file_prefix = file_prefix or self._data_source
        

        # Creating generator for training data
        train = self.cwd + f"\\mels_{file_prefix}_train"
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(train,
                                                            target_size=(
                                                                288, 432),
                                                            color_mode="rgba",
                                                            class_mode='categorical',
                                                            batch_size=batch_size)
        # Creating generator for test data
        test = self.cwd + f"\\mels_{file_prefix}_test"
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(test,
                                                          target_size=(
                                                              288, 432),
                                                          color_mode='rgba',
                                                          class_mode='categorical',
                                                          batch_size=batch_size)

        return train_generator, test_generator
      
    def convert_audio_to_numpy_array(self, audio_file_path: str):
      """ Converts audio file to np array representation of mel spectrogram

      Keyword arguments:
      audio_file_path -- Path to audio file
      """
      
      # Read audio file
      y, sr = librosa.load(audio_file_path)
      mels = librosa.feature.melspectrogram(y=y, sr=sr)
      
      # Create plot and temp save
      plt.Figure()
      plt.imshow(librosa.power_to_db(mels, ref=np.max))
      plt.savefig("temp_mel.jpg")
      
      # Read plot in as 4D numpy array
      img_data = img_to_array(load_img(r'temp_mel.jpg', 
                    target_size=(288, 432), color_mode="rgba")) / 255
      img_data = np.expand_dims(img_data, axis = 0)
      
      # Remove temp plot and return np_array
      os.remove("temp_mel.jpg")
      return img_data
    
    def predict_genre(self, audio_file_path: str):
      """ Makes genre prediciton for provided audio file, returning probability
      array

      Keyword arguments:
      audio_file_path -- Path to audio file
      """
      
      # Convert audio to np array and make prediction
      img_data = self.convert_audio_to_numpy_array(audio_file_path)
      return self.model.predict(img_data, verbose=3)
      

    def compile_model(self, loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(
                          learning_rate=0.0008, decay=0.01),
                      metrics=['categorical_accuracy']):
        """ Compiles the created genre model

        Keyword arguments:
        loss -- Type of loss function to use when compiling model
        optimizers -- Type of optimizer to use when compiling model
        metrics -- Metrics to report for each epoch
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def save_model(self, path: str = None):
        """ Saves model to directory to be loaded

        Keyword arguments:
        path -- full path to desired directory
        """
        self.model.save(path or self._model_path)

    def load_model(self, model_path: str):
        """ Loads saved model

        Keyword arguments:
        model_path -- full path to existing model directory
        """
        self.model = keras.models.load_model(model_path)
