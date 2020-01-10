import tensorflow as tf
# from keras.layers import Input, Dense, Conv2D, Flatten
# from keras.models import Model
# from keras.utils import plot_model
import numpy as np
from typing import List
import numbers


def expand_params(param, length):
    if isinstance(param, numbers.Number):
        return [param] * length
    if len(param) == 1:
        param *= length
    assert(len(param) == length)
    return param


def generate_sparse_mtl_head(task_labels: List[str],
                             number_classes: List[int],
                             layers_per_head,
                             sizes_per_head,
                             filters_per_head,
                             input_model):

    number_heads = len(task_labels)

    number_classes = expand_params(number_classes, number_heads)
    sizes_per_head = expand_params(sizes_per_head, number_heads)
    layers_per_head = expand_params(layers_per_head, number_heads)
    sizes_per_head = expand_params(sizes_per_head, number_heads)
    filters_per_head = expand_params(filters_per_head, number_heads)

    head_list = []

    for taskid, tasklabel in enumerate(task_labels):
        x = input_model.outputs[0]
        num_layers = layers_per_head[taskid]
        this_sizes = expand_params(sizes_per_head[taskid], num_layers)
        this_filters = expand_params(filters_per_head[taskid], num_layers)
        for layerid in range(num_layers):
            x = tf.keras.layers.Conv2D(filters=this_filters[layerid],
                                       kernel_size=(this_sizes[layerid], 1),
                                       padding='same',
                                       name=f"{tasklabel}_{layerid}")(x)
        x = tf.keras.layers.Flatten()(x)
        head_output = tf.keras.layers.Dense(number_classes[taskid],
                                            activation="softmax",
                                            name=f"{tasklabel}_out")(x)
        head_list.append(head_output)

    return head_list


def generate_mtl_head(task_labels: List[str],
                      neurons_per_head: List[int],
                      layers_per_head: List[int],
                      number_classes: List[int],
                      dense_dropout: List[float],
                      input_model):

    number_heads = len(task_labels)

    number_classes = expand_params(number_classes, number_heads)
    neurons_per_head = expand_params(neurons_per_head, number_heads)
    layers_per_head = expand_params(layers_per_head, number_heads)
    dense_dropout = expand_params(dense_dropout, number_heads)

    head_list = []

    fx = tf.keras.layers.Flatten()(input_model.outputs[0])

    for taskid, tasklabel in enumerate(task_labels):
        x = fx
        num_layers = layers_per_head[taskid]
        neuron_counts = neurons_per_head[taskid]
        dropout = dense_dropout[taskid]
        if isinstance(neuron_counts, int):
            neuron_counts = [neuron_counts] * num_layers
        elif len(neuron_counts) == 1:
            neuron_counts *= num_layers
        assert(len(neuron_counts) == num_layers)
        for i in range(num_layers):
            x = tf.keras.layers.Dense(neuron_counts[i], activation="relu",
                                      name=f"{tasklabel}_{i}")(x)
            if dropout[i] > 0:
                x = tf.keras.layers.Dropout(
                    dropout[i], name=f"dropout_{tasklabel}_{i}")(x)
        head_output = tf.keras.layers.Dense(
            number_classes[taskid], activation="softmax", name=f"{tasklabel}_out")(x)
        head_list.append(head_output)

    return head_list


if __name__ == "__main__":

    data = tf.keras.layers.Input(shape=(1, 1, 1,))

    task_labels = ["bigtask", "task1", "task2"]
    number_classes = [42, 1, 1]
    base_model = tf.keras.Model(inputs=data, outputs=data)
    head_list = generate_sparse_mtl_head(task_labels, number_classes,
                                         [3, 2, 2], [100, 50, 50], [64, 64, 64], base_model)

    model = tf.keras.Model(inputs=data, outputs=head_list)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file="base_mtl_head.png", show_shapes=True, dpi=320)
