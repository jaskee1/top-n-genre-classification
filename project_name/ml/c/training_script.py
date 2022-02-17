import sys
import time

import tensorflow as tf

import project_name.data.data_loader as dl
import project_name.data.feature_recorder as fr
from project_name.ml.c.ml_algo_c import MlAlgoC

# Are we loading or training from a fresh state?
LOAD_ALGO = False
# Save the trained algo?
SAVE_ALGO = False
# Are we using the resnet algo_c variant?
USE_RESNET = True

if __name__ == '__main__':

    start_time = time.time()

    data_type = 'fma'
    fma_set = 'medium'

    if len(sys.argv) > 1:
        data_type = sys.argv[1]
    if len(sys.argv) > 2:
        fma_set = sys.argv[2]

    if data_type == 'fma' and (fma_set == 'medium' or fma_set == 'large'):
        cache = False
        batch_size = 64
    else:
        cache = True
        batch_size = 32

    # Used to gather up all the files and associate them with splits and
    # labels.
    loader = dl.DataLoader(data_type=data_type, fma_set=fma_set)
    # Used to load features from .tfrecord files
    recorder = fr.FeatureRecorder()

    # Get filenames, labels, and splits
    metadata = loader.gather_data('.c.tfrecord', include_labels=False)
    # print(metadata.shape)
    # print(metadata)

    # Separate the splits
    training = metadata[metadata['split'] == 'training']
    validation = metadata[metadata['split'] == 'validation']
    test = metadata[metadata['split'] == 'test']
    # Debug --- REMOVE LATER
    print('Splits:')
    print(training.shape)
    print(validation.shape)
    print(test.shape)

    # Create smart tensorflow dataset objects that can load the data on demand
    training = MlAlgoC.create_dataset(training, batch_size=batch_size,
                                      cache=cache)
    validation = MlAlgoC.create_dataset(validation, batch_size=batch_size,
                                        cache=cache)
    test = MlAlgoC.create_dataset(test, batch_size=batch_size, cache=cache)

    # Get previously trained algo to train it some more.
    if LOAD_ALGO:
        if USE_RESNET:
            ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_RESNET_PATH)
        else:
            ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_PATH)
    # Build the algo model from scratch and train it
    else:
        # We have to retrieve 1 element (acutally 1 batch) to inspect the data
        # shape, which we need to properly set up the ML model input/output.
        for elem in training.take(1):
            input_shape = tf.shape(elem[0])[1:].numpy()
            label_shape = tf.shape(elem[1])[1:].numpy()
            # Debug --- REMOVE LATER
            print('input shape:')
            print(tf.shape(elem[0]).numpy())
            print('label shape:')
            print(label_shape)
        ml_algo = MlAlgoC(input_shape=input_shape, output_shape=label_shape,
                          use_resnet=USE_RESNET)

    # Compile the model and fit the training data
    ml_algo.compile_model()
    ml_algo.model.fit(training, epochs=40, validation_data=validation)
    # Evaluate performance on the test set
    ml_algo.model.evaluate(test)

    # Save our trained algo for future usage
    if SAVE_ALGO:
        if USE_RESNET:
            ml_algo.save_model(path=MlAlgoC.MODEL_RESNET_PATH)
        else:
            ml_algo.save_model(path=MlAlgoC.MODEL_PATH)
