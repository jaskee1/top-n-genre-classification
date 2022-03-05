from genre_muse.ml.b.MlAlgoB import MlAlgoB
from genre_muse.ml.a.ml_a import MlA
from genre_muse.ml.c.ml_algo_c import MlAlgoC
import numpy as np

class Ensemble:
    """
    Class to run prediction using ensemble
    """

    def __init__(self):
        """
        Initializes class by preparing the models
        """
        self.cnn_model_a = MlA()
        self.cnn_model_b = MlAlgoB(classes=8)
        self.cnn_model_c = MlAlgoC(load_path=MlAlgoC.MODEL_PATH)


    def predict_genres(self, audio_filepath):
        """
        Predicts the genre of a given audiofile
        Returns
        -------
`       An array of the prediction accuracies
        """
        # TODO: update weights to give model with higher accuracy more weight
        weight_a = 1
        weight_b = 1
        weight_c = 1

        predictions_a = weight_a * np.asarray(self.cnn_model_a.predict_a(audio_filepath))
        predictions_b = weight_b * np.asarray(self.cnn_model_b.predict_genre(audio_filepath))
        predictions_c = weight_c * np.asarray(self.cnn_model_c.predict_from_file(audio_filepath))

        combined_predictions = (predictions_a + predictions_b + predictions_c) / 3

        genres = ["electronic", "experimental", "folk", "hiphop", "instrumental", "international", "pop", "rock"]

        results_dictionary = {}

        for genre_index in range(0, 8):
            results_dictionary[genres[genre_index]] = combined_predictions[0][genre_index]

        return results_dictionary

