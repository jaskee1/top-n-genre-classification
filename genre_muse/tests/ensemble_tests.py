import os
import unittest


from genre_muse.ml.ensemble import Ensemble


class EnsembleTests(unittest.TestCase):
    """
    Tests for ensemble
    """

    def test_ensemble(self):
        """
        Tests that the ensemble can run without errors
        """
        filepath = os.getcwd() + '/test_audiofiles/10805.wav'
        test_ensemble = Ensemble()
        prediction = test_ensemble.predict_genres(filepath)
        print(prediction)


if __name__ == '__main__':
    unittest.main()