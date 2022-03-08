from pathlib import Path

import librosa
from tensorflow import keras
import numpy as np


class MlA:
    """
    Class for using model A
    """

    def __init__(self):
        """
        Initializes class by loading model A
        """
        # TODO: update model name when model is complete
        self.model = keras.models.load_model(Path(__file__).parent / 'model_a.h5')

    def _get_data_from_audio_file(self, filepath):
        """
        Gets the MFCC data from a given audio file
        Returns
        -------
        returns the MFCC data
        """
        y, sr = librosa.load(filepath, offset=5, duration=20)
        # TODO: update with values from working model
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, n_mfcc=20)
        mfcc = np.asarray(mfcc)
        mfcc = mfcc.reshape(-1, len(mfcc), len(mfcc[0]), 1)
        return mfcc

    def predict_a(self, filepath):
        """
        Makes a genre prediction on a given audio file
        Returns
        -------
        returns the array of predictions
        """
        features = self._get_data_from_audio_file(filepath)
        return self.model.predict(features)

