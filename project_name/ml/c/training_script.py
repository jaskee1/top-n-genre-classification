import sys
# import os
import time

from ml_c import MlAlgoC

# script_dir = os.path.dirname(__file__)
# mymodule_dir = os.path.join(script_dir, '..', '..')
# sys.path.append(mymodule_dir)

import project_name.data.data_loader as dl           # noqa: E402
import project_name.data.feature_recorder as fr      # noqa: E402

LOAD_ALGO = False

if __name__ == '__main__':

    start_time = time.time()

    data_type = 'gtzan'

    if len(sys.argv) > 1:
        data_type = sys.argv[1]
    if len(sys.argv) > 2:
        fma_set = sys.argv[2]

    loader = dl.DataLoader(data_type=data_type)
    recorder = fr.FeatureRecorder()
    ml_algo = MlAlgoC()

    file_paths = loader.gather_data('tfrecord', include_labels=False)

    # Get the splits
    training, validation, testing = ml_algo.split_dataframe_gtzan(file_paths)
    # Create smart tensorflow dataset objects that can load the data
    training = ml_algo.create_dataset(file_paths)
    validation = ml_algo.create_dataset(file_paths)
    testing = ml_algo.create_dataset(file_paths)

    # Get previously trained algo to train it some more!
    if LOAD_ALGO:
        ml_algo.load_model()

    # Compile the model and fit the training data
    ml_algo.compile_model()
    ml_algo.model.fit(training, epochs=50, validation_data=validation)
    # Evaluate performance on the test set
    ml_algo.model.evaluate(testing)
    # Save our trained algo for future usage
    ml_algo.save_model()
