import sys
import time

import tensorflow as tf

from project_name.ml.c.ml_algo_c import MlAlgoC

import project_name.data.data_loader as dl
import project_name.data.feature_recorder as fr

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

    file_paths = loader.gather_data('.c.tfrecord', include_labels=False)

    # Get the splits
    training, validation, testing = MlAlgoC.split_dataframe_gtzan(file_paths)
    # Create smart tensorflow dataset objects that can load the data on demand
    training = MlAlgoC.create_dataset(training)
    validation = MlAlgoC.create_dataset(validation)
    testing = MlAlgoC.create_dataset(testing)

    # Get previously trained algo to train it some more!
    if LOAD_ALGO:
        ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_PATH)
    # Build the algo model from scratch and train it
    else:
        # We have to retrieve 1 element (acutally 1 batch) to inspect the data
        # shape, which we need to properly set up the ML model input/output.
        for elem in training.take(1):
            input_shape = tf.shape(elem[0])[1:].numpy()
            label_shape = tf.shape(elem[1])[1:].numpy()
            print(input_shape)
        ml_algo = MlAlgoC(input_shape=input_shape, output_shape=label_shape)

    # Compile the model and fit the training data
    ml_algo.compile_model()
    ml_algo.model.fit(training, epochs=50, validation_data=validation)
    # Evaluate performance on the test set
    ml_algo.model.evaluate(testing)
    # Save our trained algo for future usage
    ml_algo.save_model()
