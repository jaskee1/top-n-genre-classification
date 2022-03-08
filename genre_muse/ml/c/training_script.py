import sys
import time

import pandas as pd
import tensorflow as tf

import genre_muse.data.data_loader as dl
from genre_muse.ml.c.ml_algo_c import MlAlgoC

# Are we loading or training from a fresh state?
LOAD_ALGO = False
# Save the trained algo?
SAVE_ALGO = True
# Are we using the resnet algo_c variant?
USE_RESNET = False
# Train with the validation set as well
# Use test set as validation at this point
COMBINE_VALIDATION = True

if __name__ == '__main__':

    start_time = time.time()

    data_type = 'prop'
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

    # Get filenames, labels, and splits
    metadata = loader.gather_data('.c.tfrecord', include_labels=False)
    # Separate the splits
    training = metadata[metadata['split'] == 'training']
    validation = metadata[metadata['split'] == 'validation']
    test = metadata[metadata['split'] == 'test']

    if COMBINE_VALIDATION:
        training = pd.concat([training, validation])
        validation = test

    # Do bagging (with replacement) on just the training set here
    training = training.sample(frac=1, replace=True, random_state=42)
    # print(metadata.shape)
    # print(metadata)
    # print(training)

    # Debug
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
            # Debug
            print('input shape:')
            print(tf.shape(elem[0]).numpy())
            print('label shape:')
            print(label_shape)
        ml_algo = MlAlgoC(input_shape=input_shape, output_shape=label_shape,
                          use_resnet=USE_RESNET)

    # Compile the model and fit the training data
    ml_algo.compile_model()
    # Print a summary for debugging purposes
    ml_algo.model.summary()
    # Using early stopping to automatically get best fit and to stop training
    # once it's found.
    earlyStop = tf.keras.callbacks.EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=7,
        restore_best_weights=True
    )
    ml_algo.model.fit(training, epochs=50, validation_data=validation,
                      callbacks=[earlyStop])

    # Evaluate performance on the test set. Note that if we're using
    # combined validation, this will just run again on the validation set,
    # which will be the same as optimal, saved results from earlier epoch
    ml_algo.model.evaluate(test)

    # Save our trained algo for future usage
    if SAVE_ALGO:
        if USE_RESNET:
            ml_algo.save_model(path=MlAlgoC.MODEL_RESNET_PATH)
        else:
            ml_algo.save_model(path=MlAlgoC.MODEL_PATH)
