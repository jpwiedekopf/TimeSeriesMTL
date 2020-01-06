import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from functools import reduce

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
# from https://www.tensorflow.org/beta/tutorials/load_data/tf_records#reading_a_tfrecord_file_2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _numpy_feature(value):
    """serializes arbitrary tensors and returns a byte_list. Use with numpy arrays."""
    return _bytes_feature(tf.io.serialize_tensor(value))


def opportunity_reader(filenames,
                       all_label_names,
                       selected_label_names,
                       number_classes,
                       x_shape=[64, 107, 1],
                       shuffle_buffer: int = 1000,
                       validation=False):
    """Read the given tfrecord files and return a parsed tf.dataset

    Arguments:
        filenames {List} -- fully qualified filenames to read into the dataset
        all_label_names {List} -- a list with all label names in the file
        selected_label_names {List} -- a list with the desired label names
        number_classes {List} -- a list with the number of classes per selected label channel

    Keyword Arguments:
        shuffle_buffer {int} -- the size of the shuffle buffer (default: {1000})

    Returns:
        tf.dataset -- the parsed dataset
    """

    def _gen_feature_description():
        """generate the feature description needed to parse the example protobuffers

        Returns:
           Dict -- the feature description
        """
        feature_description = {
            'X': tf.io.FixedLenFeature([], tf.string)
        }
        for ln in all_label_names:
            feature_description[ln] = tf.io.FixedLenFeature([], tf.int64)
        return feature_description

    @tf.function
    def _set_shape(x):
        """Set the shape of the tensor to the required shape.
        Annotated with @tf.function, so it actually works..."""
        x.set_shape(x_shape)
        return x

    def _parse_function(example_proto):
        """Parses a protobuf message into the requested format. Use with dataset.map

        Arguments:
            example_proto {str} -- The byte string containing the serialized protobuf

        Returns:
            Tuple -- A tuple (X, (Y1, Y2, ..., Yn)) with the requested channels in the Y part
        """
        parsed_features = tf.io.parse_single_example(
            example_proto, feature_description)
        X = tf.io.parse_tensor(parsed_features['X'], tf.float32)
        X = _set_shape(X)

        # tf.keras expects each generator item to consist of two items ->
        # inputs and labels.
        # each desired label channel is one-hot encoded and placed into a
        # tuple in the correct order, given by label_names
        Y = tuple(tf.one_hot(parsed_features[ln], number_classes[li])
                  for li, ln in enumerate(selected_label_names))
        return (X, Y)

    assert len(number_classes) == len(
        selected_label_names)

    feature_description = _gen_feature_description()

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    if not validation:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    return dataset


def tf_data_record_writer(bigX, bigY, label_names, label_indices, number_classes, file_path_prefix, verbose=True):
    """Writes two numpy arrays (features and labels) to tfrecord files

    Arguments:
        bigX {np.ndarray} -- The feature array
        bigY {np.ndarray} -- The label array
        label_names {List} -- A list containing human-readable names for each label column, in order
        label_indices {List} -- A list containing the label channels that should be written to file
        number_classes {List} -- The number of classes for each label channel
        file_path_prefix {str} -- The filename of the file, without .tfrecord

    Keyword Arguments:
        verbose {bool} -- Whether to output during writing (default: {True})
    """

    def _serialize_example(feat_x, feat_y, label_names):
        """Serializes the examples into a protobuf message

        Arguments:
            feat_x {np.ndarray} -- The feature of this example, arbitrary dimension
            feat_y {List[int]} -- The integer labels associated with this example
            label_names {List[str]} -- Human-readable names for each label

        Returns:
            str -- The serialized protobuf message as a byte string
        """
        #feat_x = tf.convert_to_tensor(feat_x)
        #feat_y = tf.convert_to_tensor(feat_y, dtype=tf.int32)
        feature = {
            'X': _numpy_feature(feat_x)
        }
        for li, ln in enumerate(label_names):
            feature[ln] = _int64_feature(feat_y[li])
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    result_tf_file = file_path_prefix + '.tfrecords'
    if verbose:
        print("Serializing {:d} examples into {}".format(
            bigX.shape[0], result_tf_file))
    shape = bigX[0, :, :].shape
    with tf.io.TFRecordWriter(result_tf_file) as writer:
        for idx in range(bigX.shape[0]):
            xarray = bigX[idx, :, :]
            xarray = np.reshape(xarray, (xarray.shape[0], xarray.shape[1], 1))
            yarray = bigY[idx, label_indices]
            ylist = [yarray[j] for j, _ in enumerate(label_indices)]
            serialized_example = _serialize_example(xarray, ylist, label_names)
            writer.write(serialized_example)
        print("x shape", shape)
        print()


if __name__ == "__main__":
    path = '/data/opportunity/window_labels-mtl-tf/all_sensors/64'
    files = os.listdir(path)
    filename = os.path.join(path, "S1-ADL1.tfrecords")
    dataset = tf_data_reader(  # os.path.join(path, files[0]),
        filename,
        selected_label_names=['ML_Both', 'LL_Locomotion'],
        all_label_names=['LL_Locomotion', 'HL_Activity', 'LL_Left',
                         'LL_LeftObject', 'LL_Right', 'LL_RightObject', 'ML_Both'],
        number_classes=[18, 6])

    for data in dataset.take(1):
        print(data)
    print(dataset)
