import numpy as np
import scipy.stats
import pickle
import tensorflow as tf
import os
import itertools
import random
from tqdm import tqdm

from typing import List

np.random.seed(42)
random.seed(42)


class DEAPWriter:

    @staticmethod
    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def numpy_feature(value):
        """serializes arbitrary tensors and returns a byte_list. Use with numpy arrays."""
        return DEAPWriter.bytes_feature(tf.io.serialize_tensor(value))

    def __init__(self, filenames):
        self.filenames = filenames
        self.records_processed = [0] * len(filenames)

    def get_filenames(self):
        return ", ".join(self.filenames)

    def __enter__(self):
        self.writers = [tf.io.TFRecordWriter(fn) for fn in self.filenames]
        self.writer_iterator = itertools.cycle(self.writers)
        print(f"Opened TFRecord files{', '.join(self.filenames)}")
        return self

    def __exit__(self, type, value, traceback):
        for i, t in enumerate(zip(self.filenames, self.writers)):
            filename, writer = t
            print(
                f"Closing {filename} after {self.records_processed[i]} records")
            writer.close()

    def serialize_example(self, data, labels):
        feature = {
            "X": DEAPWriter.numpy_feature(data)
        }
        for ln in labels.keys():
            feature[ln] = DEAPWriter.int64_feature(labels[ln])
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write_example(self, data, labels):
        self.shape = data.shape
        serialized = self.serialize_example(data, labels)
        this_writer = next(self.writer_iterator)
        writer_id = self.writers.index(this_writer)
        this_writer.write(serialized)
        self.records_processed[writer_id] += 1


def process_directory(
        dir_name: str,
        target_directory: str,
        out_names: List[str],
        target_splits: int,
        method: str,
        mode: str,
        trim_samples: int,
        valence_classes: int,
        arousal_classes: int,
        dominance_classes: int,
        liking_classes: int) -> None:

    out_names = [os.path.join(target_directory, on) for on in out_names]
    file_id = 0
    file_list = os.listdir(dir_name)

    if mode == 'round-robin':
        random.shuffle(file_list)
        # probably does not affect the result at all, but will not hurt
        with DEAPWriter(out_names) as writer:
            pb = tqdm(file_list, position=0, leave=True)
            for file in pb:
                if ".dat" in file:
                    full_name = os.path.join(dir_name, file)
                    pb.write(f"Processing file {full_name}")
                    process_file(pickle_file=full_name,
                                 writer=writer,
                                 target_splits=target_splits,
                                 valence_classes=valence_classes,
                                 arousal_classes=arousal_classes,
                                 trim_samples=trim_samples,
                                 file_id=file_id,
                                 method=method,
                                 dominance_classes=dominance_classes,
                                 liking_classes=liking_classes,
                                 progress_bar=pb)
                    file_id += 1
    else:
        file_list.sort()
        pb = tqdm(file_list, position=0, leave=True)
        for file in pb:
            if ".dat" in file:
                full_name = os.path.join(dir_name, file)
                out_name = out_names[file_id]
                pb.write(f"Processing file {full_name}")
                with DEAPWriter([out_name]) as writer:
                    process_file(pickle_file=full_name,
                                 writer=writer,
                                 target_splits=target_splits,
                                 valence_classes=valence_classes,
                                 arousal_classes=arousal_classes,
                                 trim_samples=trim_samples,
                                 file_id=file_id,
                                 method=method,
                                 dominance_classes=dominance_classes,
                                 liking_classes=liking_classes,
                                 progress_bar=pb)
                file_id += 1
    print(f"Finished processing DEAP directory {dir_name}")


def unpickle(file):
    with open(file, "rb") as rf:
        x = pickle.load(rf, encoding="latin1")
        data = x["data"]
        labels = x["labels"]
        return data, labels


def process_file(
        pickle_file: str,
        writer: DEAPWriter,
        method: str,
        trim_samples: int,
        file_id: int,
        target_splits: int,
        valence_classes: int,
        arousal_classes: int,
        dominance_classes: int,
        liking_classes: int,
        progress_bar,
        axes_order='sensors-first'):
    data, labels = unpickle(pickle_file)
    # it is easier to work with the data in order experiment x time x channel, so we can iterate over the time axis easier
    data = np.swapaxes(data, 1, 2)
    num_samples = data.shape[1]
    num_channels = data.shape[2]
    label_names = ["valence", "arousal", "dominance", "liking"]
    label_transformer = LabelTransformer(
        valence_classes,
        arousal_classes,
        dominance_classes,
        liking_classes,
        label_names)

    if trim_samples > 0:
        data = data[:, trim_samples:, :]

    num_experiments = data.shape[0]
    if method == "own":
        np.random.shuffle(data)
        for experiment in range(num_experiments):
            experiment_data = data[experiment, :]
            experiment_labels = labels[experiment, :]

            transformed_labels = label_transformer.transform_labels(
                experiment_labels)

            split_experiment_data = np.split(experiment_data,
                                             target_splits,
                                             axis=0)
            random.shuffle(split_experiment_data)

            for sample in split_experiment_data:
                if axes_order == 'sensors-first':
                    sample = np.swapaxes(sample, 0, 1)
                writer.write_example(sample, transformed_labels)
    elif method == "tripathi":
        statfuns = [np.mean,
                    np.median,
                    np.max,
                    np.min,
                    np.std,
                    np.var,
                    np.ptp,
                    scipy.stats.skew,
                    scipy.stats.kurtosis]

        def _run_statistics(input_array, statistics):
            return np.array([fun(input_array) for fun in statistics])

        overall = np.apply_along_axis(
            _run_statistics, 1, data, statistics=statfuns)
        progress_bar.write("Calc'd overall stats")

        slices = np.split(data, 10, axis=1)
        slice_stat = [np.apply_along_axis(_run_statistics,
                                          1,
                                          slice,
                                          statistics=statfuns) for slice in slices]
        progress_bar.write("Calc'd slice-wise stats")

        experiment = np.arange(0, num_experiments, 1).reshape(
            (num_experiments, 1))
        experiment = np.repeat(
            experiment[:, :, np.newaxis], num_channels, axis=2)
        subject = np.ones((num_experiments, 1, num_channels)) * file_id - 1

        concat = np.concatenate(
            slice_stat + [overall, experiment, subject], axis=1)

        np.random.shuffle(concat)

        for experiment in range(num_experiments):
            experiment_labels = labels[experiment, :]
            transformed_labels = label_transformer.transform_labels(
                experiment_labels)
            writer.write_example(concat[experiment, :], transformed_labels)

    else:
        raise ValueError("Unsupported method")


class LabelTransformer:

    def __init__(self,
                 valence_classes: int,
                 arousal_classes: int,
                 dominance_classes: int,
                 liking_classes: int,
                 label_names):
        self.valence_classes = self.build_classes(valence_classes)
        self.arousal_classes = self.build_classes(arousal_classes)
        self.dominance_classes = self.build_classes(dominance_classes)
        self.liking_classes = self.build_classes(liking_classes)
        self.label_names = label_names

    def build_classes(self, num_classes: int, min_value=1.0, max_value=9):
        breaks = np.linspace(min_value, max_value, num_classes + 1)
        a, b = itertools.tee(breaks)
        next(b, None)
        return list(zip(a, b))

    def transform_labels(self, labels, return_np=False):
        valence_label = self.transform_label_channel_with_classes(
            labels[0], self.valence_classes)
        arousal_label = self.transform_label_channel_with_classes(
            labels[1], self.valence_classes)
        dominance_label = self.transform_label_channel_with_classes(
            labels[2], self.valence_classes)
        liking_label = self.transform_label_channel_with_classes(
            labels[3], self.valence_classes)
        if return_np:
            return np.array([valence_label, arousal_label, dominance_label, liking_label])
        return {k: v for k, v in zip(self.label_names,
                                     [valence_label, arousal_label,
                                      dominance_label, liking_label])
                }

    def transform_label_channel_with_classes(self, label_channel, classes):
        for class_id, class_tuple in enumerate(classes):
            min_inclusive, max_exclusive = class_tuple
            if (label_channel >= min_inclusive) and (label_channel < max_exclusive):
                label_channel = class_id
                break
        # last class is including the (last) maximum value (9.0)
        if label_channel == max_exclusive:
            label_channel = len(classes) - 1
        # one_hot = LabelTransformer.onehot(len(classes), label_channel)
        # return one_hot
        # return as scalar
        return label_channel

    @staticmethod
    def onehot(n_classes, labels):
        return np.eye(n_classes)[labels]


if __name__ == "__main__":

    num_splits = 5
    num_subjects = 32

    mode = 'round-robin'

    if mode == 'round-robin':
        out_names = [
            f"deap_split{(sp + 1):02}.tfrecords" for sp in range(num_splits)]
    elif mode == 'per-subject':
        out_names = [
            f"deap_subject{(sp+1):02}.tfrecords" for sp in range(num_subjects)]

    process_directory(
        dir_name=r"/data/deap/data_preprocessed_python",
        target_directory=f"/data/deap/windowed-3s-unswapped/{mode}",
        out_names=out_names,
        mode=mode,
        method="own",
        target_splits=20,
        trim_samples=384,
        valence_classes=2,
        arousal_classes=2,
        dominance_classes=2,
        liking_classes=2
    )
