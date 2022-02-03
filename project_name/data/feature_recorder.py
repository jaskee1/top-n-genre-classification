import tensorflow as tf


class FeatureRecorder:
    """
    """

    def __init__(self, feature_descriptor=None):
        """
        """
        self.feature_descriptor = feature_descriptor

    def _make_feature(self, value):
        """
        """

        value = tf.convert_to_tensor(value)
        value = tf.io.serialize_tensor(value)
        return self._bytes_feature(value)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def packIntoProtobuf(self, raw_features):
        """
        """

        fd = self.feature_descriptor
        packed_features = {}

        for i, feature in enumerate(raw_features):
            packed_features[fd[i][0]] = self._make_feature(feature)

        return tf.train.Example(features=tf.train.Features(
            feature=packed_features))

    def write_tfrecord(self, protobuf, file_path):
        """
        """
        filename = str(file_path.with_suffix('.tfrecord'))
        with tf.io.TFRecordWriter(filename) as f:
            f.write(protobuf.SerializeToString())

    def read_tfrecord(self, file_path):
        """
        """
        filename = str(file_path.with_suffix('.tfrecord'))
        feature_des = tf.io.FixedLenFeature([], tf.string, default_value='')
        feature_description = {
                x[0]: feature_des for x in self.feature_descriptor
            }

        bundle = []
        for serialized_example in tf.data.TFRecordDataset([filename]):
            bundle.append(
                tf.io.parse_single_example(
                    serialized_example,
                    feature_description))

        for tensor_dict in bundle:
            for i, k in enumerate(feature_description.keys()):
                tensor_dict[k] = tf.io.parse_tensor(
                    tensor_dict[k],
                    self.feature_descriptor[i][1])

        # Remove the wrapping List if only one Example record is found
        if len(bundle) == 1:
            bundle = bundle[0]

        return bundle
