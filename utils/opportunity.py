import tensorflow as tf


def labelTableForRange(min_inclusive: int, max_inclusive: int):
    return {f'{k}': v for (k, v) in [(0, 0)] +
            list(zip(range(min_inclusive, max_inclusive + 1),
                     range(min_inclusive, max_inclusive + 1)))}


# Gives the association between the original labels and the labels used for the training of the network
labelsTable = [
    labelTableForRange(1, 5),  # Locomotion 1-5
    labelTableForRange(101, 105),  # HL Activity
    labelTableForRange(201, 213),  # LL Left
    labelTableForRange(301, 323),  # LL Left Object
    labelTableForRange(401, 413),  # LL Right
    labelTableForRange(501, 523),  # LL Right Object
    {'0': 0, '406516': 1, '406517': 2, '404516': 3, '404517': 4, '406520': 5, '404520': 6,
     '406505': 7, '404505': 8, '406519': 9, '404519': 10, '406511': 11, '404511': 12, '406508': 13,
     '404508': 14, '408512': 15, '407521': 16, '405506': 17}  # ML Both Arms
]

nbClasses = [len(q) for q in labelsTable]


def opportunity_num_classes_for_label_channel(channel):
    return len(labelsTable[channel])


def opportunity_select_channels_tf(channels):
    label_names = ['LL_Locomotion', 'HL_Activity', 'LL_Left',
                   'LL_LeftObject', 'LL_Right', 'LL_RightObject', 'ML_Both']
    label_names = [label_names[i] for i in channels]
    num_classes = [
        opportunity_num_classes_for_label_channel(i) for i in channels]

    return label_names, num_classes


def opportunity_select_channels_numpy(y_train, y_test, channels, args):
    label_names = ['LL_Locomotion', 'HL_Activity', 'LL_Left',
                   'LL_LeftObject', 'LL_Right', 'LL_RightObject', 'ML_Both']
    label_names = [label_names[i] for i in channels]

    num_classes = [
        opportunity_num_classes_for_label_channel(i) for i in channels]

    if args.dry_run:
        return None, None, label_names, num_classes

    y_train = [y_train[:, c] for c in channels]
    y_train = [tf.keras.utils.to_categorical(
        y, num_classes[i]) for i, y in enumerate(y_train)]
    y_test = [y_test[:, c] for c in channels]
    y_test = [tf.keras.utils.to_categorical(
        y, num_classes[i]) for i, y in enumerate(y_test)]
    return y_train, y_test, label_names, num_classes
