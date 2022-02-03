import sys
import os

import tensorflow as tf

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..')
sys.path.append(mymodule_dir)

import data.feature_recorder as fr      # noqa: E402

recorder = fr.FeatureRecorder()


def create_dataset(dataframe, shuffle_buffer_size=10000,
                   n_parse_threads=5, batch_size=32):
    # Get bytes from tfrecord files from the input dataframe
    dataset = tf.data.TFRecordDataset(dataframe['data'])
    # Map the raw bytes to the properly parsed data
    dataset = dataset.map(recorder.read_tfrecord_from_tfrecord_dataset,
                          num_parallel_calls=n_parse_threads)
    # Apply shuffling for order randomization
    dataset = dataset.shuffle(shuffle_buffer_size)
    # Get batches, with prefetching for the next batch
    return dataset.batch(batch_size).prefetch(1)
