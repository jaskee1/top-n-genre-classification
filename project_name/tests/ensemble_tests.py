import unittest
from project_name.ml.ensemble import Ensemble


class EnsembleTests(unittest.TestCase):

    def test_ensemble(self):
        """


        """
        test_ensemble = Ensemble()
        prediction = test_ensemble.predict_genres('')
        print(prediction)


if __name__ == '__main__':
    unittest.main()