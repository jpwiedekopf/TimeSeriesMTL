# from keras.models import Sequential
# from keras.models import Model
# from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling2D, Flatten, Dropout
# from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
from tensorflow import keras
from tensorflow.keras import layers, models


def gen_inputs(shape, with_batch_normalization=True):
    inputs = tf.keras.layers.Input(shape=shape, name="input")

    if with_batch_normalization:
        x = tf.keras.layers.BatchNormalization()(inputs)
    else:
        x = inputs

    return x, inputs


def normConv3(
    input_shape,
    num_kernels=[64, 32, 16],
    filter_sizes=[(1, 8), (1, 6), (1, 4)],
    pool_sizes=[(1, 4), (1, 3), (1, 2)],
    activation_conv='relu',
    units_mlp=[56],
    activation_mlp='relu',
    nb_classes=None,
    label_names=None,
    with_head=False,
    with_batch_normalization=True
):
    x, inputs = gen_inputs(input_shape, with_batch_normalization)

    for conv_id in range(len(filter_sizes)):
        conv_size = filter_sizes[conv_id]
        pool_size = pool_sizes[conv_id]
        conv_kernels = num_kernels[conv_id]

        x = tf.keras.layers.Conv2D(conv_kernels,
                                   kernel_size=conv_size,
                                   activation=activation_conv,
                                   name=f"conv{conv_id}")(x)
        if pool_size != 0:
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             name=f"pool{conv_id}")(x)

    if len(units_mlp) > 0:
        x = tf.keras.layers.Flatten()(x)
    for mlp_id in range(len(units_mlp)):
        units = units_mlp[mlp_id]
        x = tf.keras.layers.Dense(units,
                                  activation=activation_mlp,
                                  name=f"dense{mlp_id}_{units}")(x)

    if with_head:
        out = []
        for li, ln in enumerate(label_names):
            out.append(tf.keras.layers.Dense(
                nb_classes[li], activation='softmax', name=f'{ln}_out')(x))
            x = out

    return tf.keras.models.Model(inputs=inputs, outputs=x)


def mlp(
    input_shape,
    num_classes,
    label_names,
    generate_head=False,
    num_units=[2000, 1000],
    with_batch_normalization=True,
    dropout=0.4
):

    x, inputs = gen_inputs(input_shape, with_batch_normalization)

    x = tf.keras.layers.Flatten()(x)

    for i, nu in enumerate(num_units):
        x = tf.keras.layers.Dense(nu, activation='relu', name=f'dense{i}')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    if generate_head:
        out = []
        for li, ln in enumerate(label_names):
            out.append(tf.keras.layers.Dense(
                num_classes[li], activation='softmax', name=f'{ln}_out')(x))
        x = out

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def convNetDEAPTripathiReluSingleChannel(
    input_shape,
    num_classes,
    label_names,
    generate_head,
):
    return convNetDEAPTripathi(
        input_shape,
        num_classes=num_classes,
        label_names=label_names,
        generate_head=generate_head,
        conv_layers=2,
        conv_filters=[100, 100],
        conv_activations=["relu", "relu"],
        conv_shapes=[(3, 3), (3, 3)],
        pool_shape=(2, 2),
        poolconv_dropout=0.5,
        fc_neurons=128,
        fc_dropout=0.25,
        fc_activation="relu",
        output_activation="softmax",
        with_batch_normalization=False
    )


def convNetDEAPTripathi(
    input_shape,
    num_classes=[2, 2],
    label_names=["valence", "arousal"],
    conv_layers=2,
    conv_filters=[100, 100],
    conv_activations=["tanh", "tanh"],
    conv_shapes=[(3, 3), (3, 3)],
    pool_shape=(2, 2),
    poolconv_dropout=0.5,
    fc_neurons=128,
    fc_dropout=0.25,
    fc_activation="tanh",
    output_activation="softplus",
    generate_head=True,
    with_batch_normalization=False
):

    x, inputs = gen_inputs(input_shape, with_batch_normalization)

    for conv_id in range(conv_layers):
        activation = conv_activations[conv_id]
        shape = conv_shapes[conv_id]
        filters = conv_filters[conv_id]
        if activation == 'both':
            conv1 = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=shape,
                activation="tanh",
                padding="valid",
                name=f"conv_{conv_id}_tanh"
            )(x)

            conv2 = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=shape,
                activation="relu",
                padding="valid",
                name=f"conv_{conv_id}_relu"
            )(x)
            x = tf.keras.layers.Concatenate(axis=1)([conv1, conv2])
        else:
            conv_name = f"conv_{conv_id}"
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=shape,
                activation=activation,
                padding="valid",
                name=conv_name
            )(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=pool_shape)(x)
    x = tf.keras.layers.Dropout(poolconv_dropout)(x)
    x = tf.keras.layers.Flatten()(x)

    if generate_head:
        x = tf.keras.layers.Dense(
            units=fc_neurons, activation=fc_activation)(x)
        x = tf.keras.layers.Dropout(fc_dropout)(x)
        outs = [tf.keras.layers.Dense(units=nc, activation=output_activation, name=f'out_{ln}_{nc}class')(
            x) for ln, nc in zip(label_names, num_classes)]
        x = outs

    return tf.keras.Model(inputs=inputs, outputs=x)


def convNet2SPS(
        input_shape,
        nbClasses,
        label_names,
        args,
        withBatchNormalization=True
):

    x, inputs = gen_inputs(input_shape, withBatchNormalization)

    tops = [x] * len(nbClasses)
    ttn_layers = [[] for _ in range(args.numlayers)]

    # convolutional feature generators
    for layer_i in range(args.numlayers):
        for li, ln in enumerate(label_names):
            in_tensor = tops[li]
            conv_name = f"conv{layer_i}_{ln}"
            c = tf.keras.layers.Conv2D(
                filters=args.numfilters[layer_i],
                kernel_size=(args.numkerns[layer_i], 1),
                activation='relu',
                padding="valid",
                name=conv_name)(in_tensor)
            x = tf.keras.layers.MaxPooling2D(
                pool_size=(args.poolsizes[layer_i], 1),
                name=f"pool{layer_i}_{ln}")(c)
            tops[li] = x
            ttn_layers[layer_i].append(conv_name)

    for li, ln in enumerate(label_names):
        in_tensor = tops[li]
        tops[li] = tf.keras.layers.Flatten(
            name=f"flatten_{ln}")(in_tensor)

    # dense units on top of convolutional feature generators
    for dense_i in range(args.numdenselayers):
        for li, ln in enumerate(label_names):
            in_tensor = tops[li]
            x = tf.keras.layers.Dense(
                units=args.numdenseunits[dense_i],
                activation='relu',
                name=f"dense{dense_i}_{ln}")(in_tensor)
            tops[li] = x

    # final outputs on top of dense classifiers
    for li, ln in enumerate(label_names):
        in_tensor = tops[li]
        out = tf.keras.layers.Dense(
            units=nbClasses[li],
            activation="softmax",
            name=f"out_{ln}")(in_tensor)
        tops[li] = out

    model = tf.keras.Model(inputs=inputs, outputs=tops)

    return model, ttn_layers


def norm_conv3_crossstitch(
    input_shape,
    nb_classes=None,
    label_names=None,
    num_kernels=[64, 32, 16],
    filter_sizes=[(1, 8), (1, 6), (1, 4)],
    pool_sizes=[(1, 4), (1, 3), (1, 2)],
    cross_stitch_after_layer=[True, True, True],
    activation_conv='relu',
    units_mlp=[56],
    activation_mlp='relu',
    with_batch_normalization=True
):
    x, inputs = gen_inputs(input_shape, with_batch_normalization)
    tops = [x] * len(label_names)

    for layer_i in range(len(num_kernels)):
        layer_tensors = []
        filters = num_kernels[layer_i]
        kernel_size = filter_sizes[layer_i]
        pool_size = pool_sizes[layer_i]
        for li, ln in enumerate(label_names):
            in_tensor = tops[li]
            conv_name = f'conv_{layer_i}_{ln}'
            pool_name = f'pool_{layer_i}_{ln}'
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation_conv,
                padding='valid',
                name=conv_name)(in_tensor)
            if pool_size is not None:
                x = layers.MaxPooling2D(
                    pool_size=pool_size,
                    name=pool_name
                )(x)
            tops[li] = x
            layer_tensors.append(x)

        if cross_stitch_after_layer[li]:
            cross_stitch_name = f'cs_{layer_i}'
            cs = CrossStitch(
                len(label_names),
                name=cross_stitch_name)(layer_tensors)
            # HACK
            unstack_name = f'cs_unstack_{layer_i}'
            tops = tf.unstack(cs, axis=0, name=unstack_name)

    for li, ln in enumerate(label_names):
        in_tensor = tops[li]
        tops[li] = layers.Flatten(
            name=f"flatten_{ln}")(in_tensor)

    for dense_i in range(len(units_mlp)):
        units = units_mlp[dense_i]
        for li, ln in enumerate(label_names):
            dense_name = f'dense_{dense_i}_{ln}'
            in_tensor = tops[li]
            x = layers.Dense(
                units=units,
                activation=activation_mlp,
                name=dense_name)(in_tensor)
            tops[li] = x

    for li, ln in enumerate(label_names):
        in_tensor = tops[li]
        out = layers.Dense(
            units=nb_classes[li],
            activation="softmax",
            name=f"out_{ln}")(in_tensor)
        tops[li] = out

    model = tf.keras.Model(inputs=inputs, outputs=tops)

    return model


def norm_conv3_sps(
    input_shape,
    nb_classes=None,
    label_names=None,
    num_kernels=[64, 32, 16],
    filter_sizes=[(1, 8), (1, 6), (1, 4)],
    pool_sizes=[(1, 4), (1, 3), (1, 2)],
    cross_stitch_after_layer=[True, True, True],
    activation_conv='relu',
    units_mlp=[56],
    activation_mlp='relu',
    with_batch_normalization=True
):
    x, inputs = gen_inputs(input_shape, with_batch_normalization)
    tops = {ln: x for ln in label_names}
    ttn_layers = [[] for _ in range(len(filter_sizes))]

    for layer_i in range(len(num_kernels)):
        filters = num_kernels[layer_i]
        kernel_size = filter_sizes[layer_i]
        pool_size = pool_sizes[layer_i]
        for ln in label_names:
            in_tensor = tops[ln]
            conv_name = f"conv_{layer_i}_{ln}"
            pool_name = f"pool_{layer_i}_{ln}"
            c = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='valid',
                activation=activation_conv,
                name=conv_name
            )(in_tensor)
            x = layers.MaxPooling2D(
                pool_size=pool_size,
                name=pool_name
            )(c)
            tops[ln] = x
            ttn_layers[layer_i].append(conv_name)

    for ln in label_names:
        in_tensor = tops[ln]
        tops[ln] = layers.Flatten(
            name=f"flatten_{ln}"
        )(in_tensor)

    for dense_i in range(len(units_mlp)):
        units = units_mlp[dense_i]
        for ln in label_names:
            dense_name = f"dense_{dense_i}_{ln}"
            in_tensor = tops[ln]
            x = layers.Dense(
                units=units,
                activation=activation_mlp,
                name=dense_name
            )(in_tensor)
            tops[ln] = x

    for li, ln in enumerate(label_names):
        in_tensor = tops[ln]
        out = layers.Dense(
            units=nb_classes[li],
            activation="softmax",
            name=f"out_{ln}")(in_tensor)
        tops[ln] = out

    model = models.Model(inputs=inputs, outputs=tops)

    return model, ttn_layers


def convNet2CrossStitch(
    input_shape,
    nbClasses,
    label_names,
    args,
    withBatchNormalization=True
):

    x, inputs = gen_inputs(input_shape, withBatchNormalization)

    tops = [x] * len(label_names)

    for layer_i in range(args.numlayers):
        layer_tensors = []

        for li, ln in enumerate(label_names):
            in_tensor = tops[li]
            conv_name = f"conv{layer_i}_{ln}"
            c = tf.keras.layers.Conv2D(
                filters=args.numfilters[layer_i],
                kernel_size=(args.numkerns[layer_i], 1),
                activation='relu',
                padding='valid',
                name=conv_name)(in_tensor)
            x = tf.keras.layers.MaxPooling2D(
                pool_size=(args.poolsizes[layer_i], 1),
                name=f"pool{layer_i}_{ln}")(c)
            tops[li] = x
            layer_tensors.append(x)

        cross_stitch_name = f"cross_stitch_{layer_i}"
        cs = CrossStitch(
            len(label_names), name=cross_stitch_name)(layer_tensors)

        # THIS IS A HACK!
        tops = tf.unstack(cs, axis=0)

    for li, ln in enumerate(label_names):
        in_tensor = tops[li]
        tops[li] = tf.keras.layers.Flatten(
            name=f"flatten_{ln}")(in_tensor)

    # dense units on top of convolutional feature generators
    for dense_i in range(args.numdenselayers):
        for li, ln in enumerate(label_names):
            in_tensor = tops[li]
            x = tf.keras.layers.Dense(
                units=args.numdenseunits[dense_i],
                activation='relu',
                name=f"dense{dense_i}_{ln}")(in_tensor)
            tops[li] = x

    # final outputs on top of dense classifiers
    for li, ln in enumerate(label_names):
        in_tensor = tops[li]
        out = tf.keras.layers.Dense(
            units=nbClasses[li],
            activation="softmax",
            name=f"out_{ln}")(in_tensor)
        tops[li] = out

    model = tf.keras.Model(inputs=inputs,
                           outputs=tops)

    return model


def convNet2Sparseish(
    input_shape,
    nkerns=[50, 40, 30],
    filterSizes=[11, 10, 6],
    activationConv='relu',
    dropout=0.4,
    withHead=False,
    cnnPadding="valid",
    withBatchNormalization=True,
    nbClasses=None
):
    add_flatten = not withHead
    return convNet2(
        inputShape=input_shape,
        nkerns=nkerns,
        filterSizes=filterSizes,
        activationConv=activationConv,
        dropout=dropout,
        cnnPadding=cnnPadding,
        withBatchNormalization=withBatchNormalization,
        withHead=withHead,
        add_flatten_on_top=add_flatten,
        activationMLP='relu',
        neuronsMLP=[1000],
        nbClasses=nbClasses
    )


def convNetDeapFFT(
    inputShape,
    nbClasses=None,
    nkerns=[96, 64],
    filterSizes=[(2, 2), (2, 2)],
    poolSizes=[2, 2],
    activationConv='relu',
    neuronsMLP=[1000],
    activationMLP='relu',
    dropout=0.4,
    withHead=False,
    cnnPadding='valid',
    withBatchNormalization=True,
    add_flatten_on_top=False
):

    data_format = 'channels_first'
    x, inputs = gen_inputs(inputShape, withBatchNormalization)

    for i in range(len(nkerns)):
        kernel_size = filterSizes[i]
        pool_size = poolSizes[i]
        print("using kernel size", kernel_size)
        print("using pool size", pool_size)
        x = tf.keras.layers.Conv2D(filters=nkerns[i],
                                   kernel_size=kernel_size,
                                   activation=activationConv,
                                   padding=cnnPadding,
                                   data_format=data_format
                                   )(x)
        if pool_size > 1:
            x = tf.keras.layers.MaxPooling2D(
                pool_size=pool_size,
                data_format=data_format
            )(x)

    out = mlpPart(
        dropout=dropout,
        activationMLP=activationMLP,
        withHead=withHead,
        neuronsMLP=neuronsMLP,
        nbClasses=nbClasses,
        x=x
    )

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def mlpPart(
    dropout,
    activationMLP,
    withHead,
    neuronsMLP,
    nbClasses,
    x
):
    if len(neuronsMLP) >= 1:
        x = tf.keras.layers.Flatten()(x)
        for i in range(len(neuronsMLP)):
            x = tf.keras.layers.Dense(
                units=neuronsMLP[i], activation=activationMLP)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

    if withHead:
        out = tf.keras.layers.Dense(
            units=nbClasses[0], activation='softmax', name='out')(x)
    else:
        out = x

    return out


def convNet2(
        inputShape,
        nbClasses=None,
        nkerns=[50, 40, 30],
        filterSizes=[11, 10, 6],
        poolSizes=[2, 3, 1],
        activationConv='relu',
        neuronsMLP=[1000, 1000, 1000],
        activationMLP='relu',
        dropout=0.4,
        withHead=False,
        cnnPadding="valid",
        withBatchNormalization=True,
        add_flatten_on_top=False,
        axes_order="time-first"):

    assert(len(filterSizes) == len(nkerns))
    assert(len(filterSizes) == len(poolSizes))

    x, inputs = gen_inputs(inputShape, withBatchNormalization)

    for i in range(len(nkerns)):
        # if (i == 0):
        #    x = Conv2D#(filters=nkerns[i],
        #    kernel_size=#(filterSizes[i], 1),
        #    activation=activationCo#nv,
        #    input_shape=inputShape)#(x)
        # else:
        if axes_order == 'time-first':
            kernel_size = (filterSizes[i], 1)
            pool_size = (poolSizes[i], 1)
        else:
            kernel_size = (1, filterSizes[i])
            pool_size = (1, poolSizes[i])
        print("using kernel size", kernel_size)
        print("using pool size", pool_size)
        x = tf.keras.layers.Conv2D(filters=nkerns[i],
                                   kernel_size=kernel_size,
                                   activation=activationConv,
                                   padding=cnnPadding)(x)
        if (poolSizes[i] > 0):
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)

    out = mlpPart(
        dropout=dropout,
        activationMLP=activationMLP,
        withHead=withHead,
        neuronsMLP=neuronsMLP,
        nbClasses=nbClasses,
        x=x
    )

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def cnnSparse(inputShape, withHead=False, nbClasses=None):
    return convNet2(inputShape,
                    nkerns=[50, 40, 30, 30],
                    filterSizes=[11, 11, 11, 7],
                    poolSizes=[2, 0, 0, 1],
                    neuronsMLP=[],
                    withHead=withHead,
                    nbClasses=nbClasses
                    )


class CrossStitch(tf.keras.layers.Layer):

    """Cross-Stitch implementation according to arXiv:1604.03539
    Implementation adapted from https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning"""

    def __init__(self, num_tasks, *args, **kwargs):
        """initialize class variables"""
        self.num_tasks = num_tasks
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        """initialize the kernel and set the instance to 'built'"""
        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.num_tasks,
                                             self.num_tasks),
                                      initializer='identity',
                                      trainable=True)
        super(CrossStitch, self).build(input_shape)

    def call(self, xl):
        """
        called by TensorFlow when the model gets build. 
        Returns a stacked tensor with num_tasks channels in the 0 dimension, 
        which need to be unstacked.
        """
        if (len(xl) != self.num_tasks):
            # should not happen
            raise ValueError()

        out_values = []
        for this_task in range(self.num_tasks):
            this_weight = self.kernel[this_task, this_task]
            out = tf.math.scalar_mul(this_weight, xl[this_task])
            for other_task in range(self.num_tasks):
                if this_task == other_task:
                    continue  # already weighted!
                other_weight = self.kernel[this_task, other_task]
                out += tf.math.scalar_mul(other_weight, xl[other_task])
            out_values.append(out)
        # HACK!
        # unless we stack, and then unstack the tensors, TF (2.0.0) can't follow
        # the graph, so it aborts during model initialization.
        return tf.stack(out_values, axis=0)

    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

    def get_config(self):
        """implemented so keras can save the model to json/yml"""
        config = {
            "num_tasks": self.num_tasks
        }
        base_config = super(CrossStitch, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


if __name__ == '__main__':
    model = norm_conv3_crossstitch(
        input_shape=(40, 128, 1),
        args=None,
        nb_classes=[2, 2],
        label_names=['v', 'a']
    )

    model.summary()

    outpath = "/home/yoshi/cs_normconv3.json"
    with open(outpath, "w") as jf:
        model_json = model.to_json(indent=2)
        jf.write(model_json)
