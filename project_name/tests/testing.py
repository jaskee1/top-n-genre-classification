# -*- coding: utf-8 -*-
import sys
import os
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..')
sys.path.append(mymodule_dir)
from data.gtzan_utils import gtzan_utils
from ml.b.MlAlgoB import MlAlgoB

# Step 1: Create Mel Spectrograms from GTZAN audio clips using utility class
gtzan = gtzan_utils()
# gtzan.buildMelSpectrograms() # Skipping this step as GTZAN is ~1.2 gb
gtzan.createTrainTestSamples() # Creating train/test from mel spectrograms included


# Step 2: Training and saving model using TF and Keras
ml_b = MlAlgoB()
training_gen, testing_gen = ml_b.create_data_generators_gtzan()
ml_b.compile_model()
ml_b.model.fit(training_gen, epochs=70, validation_data=testing_gen)
ml_b.save_model()