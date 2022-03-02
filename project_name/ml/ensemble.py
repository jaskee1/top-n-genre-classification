from project_name.ml.b.MlAlgoB import MlAlgoB
from project_name.ml.c import ml_algo_c
from project_name.ml.a.ml_a import MlA
from project_name.ml.c.ml_algo_c import MlAlgoC


class Ensemble:
    """

    """

    def __init__(self):
        """
        Initializes class by preparing the models
        """
        # self.cnn_model_a = MlA()
        # self.cnn_model_b = MlAlgoB()
        self.cnn_model_c = MlAlgoC(load_path=MlAlgoC.MODEL_PATH)


    def predict_genres(self, audio_filepath):
        """
        Predicts the genre of a given audiofile
        Returns
        -------
`       An array of the prediction accuracies
        """
        weight_a = 1
        weight_b = 1
        weight_c = 1

        predictions_a = weight_a * self.cnn_model_a.predict_a(audio_filepath)

        predictions_b = weight_b * self.cnn_model_b.predict_genre(audio_filepath)

        features = self.cnn_model_c.prep_data_from_file(audio_filepath)
        predictions_c = weight_c * self.cnn_model_c.predict(features)

        combined_predictions = (predictions_a + predictions_b + predictions_c) / 3

        genres = ["electronic", "experimental", "folk", "hiphop", "instrumental", "international", "pop", "rock"]

        results_dictionary = {}

        for genre_index in range(0, 8):
            results_dictionary[genres[genre_index]] = combined_predictions[genre_index]

        return results_dictionary

