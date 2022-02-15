from pathlib import Path
import tensorflow as tf


class FeatureRecorder:
    """
    Provides methods for recording features, or other data, into
    .tfrecord files and/or reading this data from .tfrecord files.

    Methods
    -------
    packIntoProtobuf(raw_features)
        Pack a list of raw features into a protobuf. Each feature in the list
        can be its own list/array/etc.
    write_tfrecord(protobufs, file_path, extension)
        Write the input protobufs into a single .tfrecord file.
    read_tfrecord(file_path, extension)
        Read a .tfrecord file and retrieve the contents. Used to
        get all contents of a single file.
    parse_example(self, serialized_example)
        Parse a single serialized example read from a .tfrecord file.
        Intended usage is for .map() with a tf.data.TFRecordDataset().
    """

    FEATURE_DESC_DEF = [('features', tf.float16), ('labels', tf.int32)]

    def __init__(self, feature_descriptor=FEATURE_DESC_DEF):
        """
        Parameters
        ----------
        feature_descriptor : list[tuples], optional
            Description of the labels and data types for the data
            you want to record. This corresponds to what you will pack
            into protobufs. For example,
                [('features', tf.float16), ('labels', tf.int32)]
        """
        self._feature_descriptor = feature_descriptor

        feature_des = tf.io.FixedLenFeature([], tf.string, default_value='')
        self._feature_description = {
                x[0]: feature_des for x in self._feature_descriptor
            }

    def _make_feature(self, value):
        """
        Make necessary conversions to a value and finally convert
        it to a tf.train.Feature, ready to be packed into a protobuf.

        Parameters
        ----------
        value : ?
            the value to be converted into a Feature

        Returns
        ------
        tf.train.Feature
            a Feature containing a tf.train.BytesList of the value.
        """
        value = tf.convert_to_tensor(value)
        value = tf.io.serialize_tensor(value)
        return self._bytes_feature(value)

    def _bytes_feature(self, value):
        """
        Convert a value to a tf.train.Feature containing a tf.train.BytesList
        representing the value.

        BytesList is used for everything so the function is more usable
        with any kind of data. Hopefully it works across various types,
        but it has not been extensively tested, so beware!

        Parameters
        ----------
        value : ?
            the value to be converted into a Feature

        Returns
        ------
        tf.train.Feature
            a Feature containing a tf.train.BytesList of the value.
        """
        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def packIntoProtobuf(self, raw_features):
        """
        Pack a list of raw features into a protobuf. Each feature in the list
        can be its own list/array/etc.

        Parameters
        ----------
        raw_features : list[?]
            Contains all the raw features that need to be packed into the
            protobuf. For example, [features, label], where features is a
            numpy array and label is a Python list.

        Returns
        ------
        tf.train.Example
            a tensorflow Example protobug, ready to be written to a 
            .tfrecord file.
        """

        fd = self._feature_descriptor
        packed_features = {}

        for i, feature in enumerate(raw_features):
            packed_features[fd[i][0]] = self._make_feature(feature)

        return tf.train.Example(features=tf.train.Features(
            feature=packed_features))

    def write_tfrecord(self, protobufs, file_path, extension):
        """
        Write the input protobufs into a single .tfrecord file.

        Parameters
        ----------
        protobufs : list[tf.train.Example]
            All protobufs that should be written to the file
        file_path
            The path to write the .tfrecord file, including the file's
            name, but NOT including the extension.
        extension
            The extension to write the file with, which can be used
            to organize or differentiate data if needed. Should include
            the dot before the extension. For example, '.c.tfrecord'.

        Returns
        ------
        none
        """
        # Use a Path object to make use of the .with_suffix method.
        filename = str(Path(file_path).with_suffix(extension))
        with tf.io.TFRecordWriter(filename) as f:
            for protobuf in protobufs:
                f.write(protobuf.SerializeToString())

    def read_tfrecord(self, file_path, extension):
        """
        Read a .tfrecord file and retrieve the contents. Used to
        get all contents of a single file.

        Parameters
        ----------
        file_path
            The path to read the .tfrecord file, including the file's
            name, but NOT including the extension.
        extension
            The extension to read the file with.

        Returns
        ------
        list[?]
            a list where each entry is the parsed contents of a protobuf
            read from the .tfrecord file.
        """
        # Use a Path object to make use of the .with_suffix method.
        filename = str(Path(file_path).with_suffix(extension))

        bundle = []
        for serialized_example in tf.data.TFRecordDataset([filename]):
            bundle.append(self.parse_example(serialized_example))

        return bundle

    @tf.function
    def parse_example(self, serialized_example):
        """
        Parse a single serialized example read from a .tfrecord file.
        Intended usage is for .map() with a tf.data.TFRecordDataset().

        Parameters
        ----------
        serialized_example
            Raw bytes that should represent a tf.train.Example protobuf
            object, such as those from using tf.data.TFRecordDataset() to
            load properly written .tfrecord files.

        Returns
        ------
        list[?]
            a list where the entries are the parsed contents of a protobuf
            read from the .tfrecord file. For example, if the protobuf
            contained 3 separate data fields, the list will have 3 elements.
        """

        tensor_dict = tf.io.parse_single_example(serialized_example,
                                                 self._feature_description)

        for i, k in enumerate(self._feature_description.keys()):
            tensor_dict[k] = tf.io.parse_tensor(
                tensor_dict[k], self._feature_descriptor[i][1])

        return tensor_dict.values()
