import tensorflow as tf
from typing import List, Dict, Tuple
import os
import json


def deap_reader(
        deap_path: str,
        selected_feature_names: List[str],
        batch_size: int = 64,
        use_one_hot=True,
        shuffle_buffer: int = 1000) -> Tuple[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:

    def _gen_feature_description() -> Dict[str, tf.io.FixedLenFeature]:
        feature_description = {
            "X": tf.io.FixedLenFeature([], tf.string)
        }
        for fn in all_feature_names:
            feature_description[fn] = tf.io.FixedLenFeature([], tf.int64)

        return feature_description

    @tf.function
    def _set_shape(x, x_shape):
        """Set the shape of the tensor to the required shape.
        Annotated with @tf.function, so it actually works..."""
        x.set_shape(x_shape)
        return x

    def _parse_function(example_proto) -> Tuple:
        parsed = tf.io.parse_single_example(
            example_proto, feature_description
        )
        X = tf.io.parse_tensor(parsed["X"], tf.float64)
        if target_shape[-1] == 1:
            X.set_shape(target_shape[:-1])
            X = tf.expand_dims(X, -1)
        else:
            X.set_shape(target_shape)
        if use_one_hot:
            Y = tuple(tf.one_hot(parsed[ln], number_classes[li])
                      for li, ln in enumerate(selected_feature_names))
        else:
            Y = tuple(parsed[ln] for ln in selected_feature_names)

        return (X, Y)

    config = read_config(deap_path)
    all_feature_names = config["meta"]["label_names"]

    feature_description = _gen_feature_description()

    train_files = config["files"]["train"]
    test_files = config["files"]["test"]

    # train_files = [fn for i, fn in enumerate(
    #     filenames) if i+1 not in validation_indices]
    # test_files = [fn for i, fn in enumerate(
    #     filenames) if i+1 in validation_indices]

    train_dataset = tf.data.TFRecordDataset(train_files)
    test_dataset = tf.data.TFRecordDataset(test_files)
    val_dataset = tf.data.TFRecordDataset(test_files)

    if len(config["shape"]) == 2:
        target_shape = tuple(config["shape"] + [1])
    else:
        target_shape = tuple(config["shape"])
    number_classes = config["meta"]["num_classes"]

    train_dataset = train_dataset.map(_parse_function)
    test_dataset = test_dataset.map(_parse_function)
    val_dataset = val_dataset.map(_parse_function)

    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer)
    test_dataset = test_dataset.shuffle(buffer_size=shuffle_buffer)

    train_dataset = train_dataset.repeat()
    # test_dataset = test_dataset.repeat()

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # if deap_mode == "per-subject":
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    # elif deap_mode == "round-robin":
    #    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    # val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, (test_dataset, val_dataset), config


def read_config(deap_path):
    json_path = os.path.join(deap_path, "meta.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError("Need a meta.json file in the data dir!")
    with open(json_path, "r") as jf:
        return json.load(jf)


if __name__ == "__main__":
    path = r'/data/deap/windowed'
    filenames = [os.path.join(
        path, f"deap_split{i+1}.tfrecord") for i in range(5)]
    validation_indices = [1]
    all_feature_names = ["valence", "arousal", "dominance", "liking"]
    number_classes = [2, 2, 2, 2]
    selected_feature_names = ['valence', 'arousal']

    sample_length = 8064 / 63
    target_shape = (sample_length, 40)

    train_dataset, test_dataset = deap_reader(
        filenames=filenames,
        validation_indices=validation_indices,
        all_feature_names=all_feature_names,
        number_classes=number_classes,
        selected_feature_names=selected_feature_names,
        target_shape=target_shape
    )

    for d in train_dataset.take(1):
        print(d)
