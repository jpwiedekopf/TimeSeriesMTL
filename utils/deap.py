import os
import tensorflow as tf
from typing import Tuple, List, Dict
from dataio.deap.deap_reader import deap_reader
import json

deap_num_classes = {
    "valence": 2,
    "arousal": 2,
    "dominance": 2,
    "liking": 2
}

deap_label_names = [
    "valence",
    "arousal",
    "dominance",
    "liking"
]


def select_channels_deap(channels):

    selected_names = [deap_label_names
                      [i] for i in channels]
    nc = [
        deap_num_classes[ln] for ln in selected_names]
    return selected_names, nc, deap_label_names


def load_deap_data(args,
                   selected_names: List[str],
                   # num_classes: List[int],
                   # all_names: List[str],
                   # target_shape: Tuple[int, ...] = (128, 40, 1)
                   ) -> Tuple[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset], Tuple, Dict]:
    path = args.deap_path
    #files = os.listdir(path)
    #files = sorted([os.path.join(path, x) for x in files if "tfrecords" in x])
    data_train, data_test, config = deap_reader(
        deap_path=args.deap_path,
        # validation_indices=args.deap_validation,
        # all_feature_names=all_names,
        # number_classes=num_classes,
        batch_size=args.batch,
        # deap_mode=args.deap_mode,
        # target_shape=target_shape,
        selected_feature_names=selected_names,
        use_one_hot=args.deap_one_hot)
    for x, y in data_train.take(1):
        shape = (x.shape[1], x.shape[2], x.shape[3])
    print("data_train:   ", data_train)
    print("data_test:    ", data_test)
    print("dataset meta: ", json.dumps(config))
    return data_train, data_test, shape, config
