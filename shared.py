from utils.utilitary_mtl import fmeasure
from os import path
from dataio.opportunity.opportunity_adapter import opportunity_reader
from utils.opportunity import opportunity_select_channels_tf
import os
import tensorflow as tf
from typing import Tuple


def load_opportunity_data_mtl_tf(dataPath: str, args):

    label_names, num_classes = opportunity_select_channels_tf(args.labels)

    print(f"Loading OPPORTUNITY dataset from {dataPath}")
    training_file_criteria = ["Drill", "ADL1", "ADL2", "ADL3"]
    test_file_criteria = ["ADL4", "ADL5"]
    training_files = []
    test_files = []
    filelist = os.listdir(dataPath)
    for fn in filelist:
        if not fn.find(".tfrecords"):
            continue
        is_train = any([fn.find(c) > 0 for c in training_file_criteria])
        is_test = any([fn.find(c) > 0 for c in test_file_criteria])
        if is_train and not is_test:
            training_files.append(path.join(dataPath, fn))
        elif not is_train and is_test:
            test_files.append(path.join(dataPath, fn))
        else:
            raise AttributeError(
                'test file should be both train and test file. illegal')

    all_label_names, _ = opportunity_select_channels_tf(list(range(7)))

    train_dataset = opportunity_reader(
        training_files,
        all_label_names=all_label_names,
        selected_label_names=label_names,
        number_classes=num_classes,
        shuffle_buffer=args.shuffle_buffer)

    val_dataset = opportunity_reader(
        test_files,
        all_label_names=all_label_names,
        selected_label_names=label_names,
        number_classes=num_classes,
        validation=True)

    train_dataset = train_dataset.repeat().batch(args.batch)
    val_dataset = val_dataset.batch(args.batch)

    for x, _ in train_dataset.take(1):
        input_shape = (x.shape[1], x.shape[2], 1)

    return train_dataset, val_dataset, input_shape, label_names, num_classes


def generate_metrics_dict(label_names, num_classes, args, dataset='opportunity', out_format_string="{ln}_out"):

    if dataset == 'opportunity':
        if len(label_names) == 1:
            return [tf.keras.metrics.CategoricalAccuracy(),
                    fmeasure]

        metrics = {}
        for _, ln in enumerate(label_names):
            outname = out_format_string.format(ln=ln)
            metrics[outname] = [tf.keras.metrics.CategoricalAccuracy(),
                                fmeasure
                                ]
        return metrics
    elif dataset == 'deap':
        one_hot = args.deap_one_hot

        def accuracy(y_true, y_pred):
            if one_hot:
                return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
            else:
                return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        fun = accuracy
        # if args.deap_one_hot:
        #     #fun = 'accuracy'
        #     fun = accuracy
        #     #fun = tf.keras.metric.CategoricalAccuracy
        # else:
        #     #fun = 'accuracy'
        #     fun = accuracy
        #fun = tf.keras.metrics.SparseCategoricalAccuracy
        if len(label_names) == 1:
            return [fun]
        else:
            metrics = {}
            for _, ln in enumerate(label_names):
                outname = out_format_string.format(ln=ln)
                metrics[outname] = [fun]
            return metrics


def verify_optimizer_args(optimizer_args, allowed_optimizers) -> Tuple[bool, str]:
    if "name" not in optimizer_args:
        return False, "no optimizer name"
    if optimizer_args["name"].lower() not in allowed_optimizers:
        return False, "invalid optimizer name"
    if "kwargs" not in optimizer_args:
        return False, "no kwargs"
    if not isinstance(optimizer_args["kwargs"], dict):
        return False, "kwargs must be a dict"
    return True, None


def build_optimizer(optimizer_args):
    implemented_optimizers = {
        "adadelta": tf.keras.optimizers.Adadelta,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adam": tf.keras.optimizers.Adam,
        "adamax": tf.keras.optimizers.Adamax,
        "ftrl": tf.keras.optimizers.Ftrl,
        "nadam": tf.keras.optimizers.Nadam,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd": tf.keras.optimizers.SGD
    }
    if optimizer_args is None:
        return tf.keras.optimizers.Adadelta(learning_rate=1.0)
    else:
        is_verified, ver_mess = verify_optimizer_args(
            optimizer_args, implemented_optimizers.keys())
        if not is_verified:
            raise ValueError(ver_mess)

        this_optimizer_name = optimizer_args["name"].lower()
        this_optimizer = implemented_optimizers[this_optimizer_name]
        kwargs = optimizer_args["kwargs"]
        return this_optimizer(**kwargs)
    raise RuntimeError()
