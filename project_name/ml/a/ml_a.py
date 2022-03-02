import librosa
from tensorflow import keras


class MlA:
    """
    Class for using model A
    """

    def __init__(self):
        """
        Initializes class by loading model A
        """
        self.model = keras.models.load_model('model_a_50.h5')

    def _get_data_from_audio_file(self, filepath):
        """
        Gets the MFCC data from a given audio file
        Returns
        -------
        returns the MFCC data
        """
        y, sr = librosa.load(filepath, offset=5, duration=20)
        return librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, n_mfcc=100)

    def predict_a(self, filepath):
        """
        Makes a genre prediction on a given audio file
        Returns
        -------
        returns the array of predictions
        """
        features = self._get_data_from_audio_file(filepath)
        return self.model.predict(features)

