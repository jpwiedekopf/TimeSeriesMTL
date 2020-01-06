import os
import argparse
import tensorflow as tf

import numpy as numpy
import sys
import time
from os import path
import shutil

from models import base_models_mtl
from utils.utilitary_mtl import fmeasure
from utils.generate_mtl_head import generate_mtl_head, generate_sparse_mtl_head

from dataio.opportunity.opportunity_adapter import opportunity_reader
from utils.opportunity import opportunity_select_channels_tf, opportunity_num_classes_for_label_channel
from shared import load_opportunity_data_mtl_tf

from dataio.deap.deap_reader import deap_reader
from utils.deap import select_channels_deap, load_deap_data

from utils.f_scores import F2Score
from evaluation import Evaluator, EvaluationCallback
from utils.app_hps import parse_args
from utils.profile import ProfileCallback
from utils.misc import Unbuffered, print_arguments

import shared
import jsons


model_choices = [
    "cnn2",
    "cnnDeapFFT",
    "cnn-sparse",
    "tripathi",
    "cnn2-sparseish",
    "mlp",
    "normConv3"
]


def generate_loss_dict(label_names, args):
    if args.dataset == 'deap' and not args.deap_one_hot:
        fun = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        fun = tf.keras.losses.CategoricalCrossentropy()
    if len(label_names) == 1:
        return [fun]
    return {f"{ln}_out": fun for ln in label_names}


def generate_class_weights(label_names, num_classes, null_weight=1.0, non_null_weight=1.0):
    if null_weight == 1.0 and non_null_weight == 1.0:
        return None
    cw = {f"{ln}_out": {} for ln in label_names}
    for li, ln in enumerate(label_names):
        lna = f"{ln}_out"
        nc = num_classes[li]
        for c in range(nc):
            if c == 0:
                cw[lna][c] = null_weight
            else:
                cw[lna][c] = non_null_weight

    return cw


def build_mtl_model(args, input_shape, label_names, nb_classes):

    print(f"Building model of type {args.model} for {len(label_names)} tasks")
    axes_order = "time-first" if args.dataset == 'opportunity' else "sensors-first"
    print("axes order", axes_order)
    if args.head_layout == 'none' and len(args.labels) == 1:
        if args.model == 'cnnDeapFFT':
            base_model = base_models_mtl.convNetDeapFFT(
                inputShape=input_shape,
                withHead=True,
                nbClasses=nb_classes)
        elif args.model == 'normConv3':
            base_model = base_models_mtl.normConv3(
                input_shape=input_shape,
                with_head=True,
                nb_classes=nb_classes,
                label_names=label_names
            )
        elif args.model == 'cnn2':
            base_model = base_models_mtl.convNet2(inputShape=input_shape,
                                                  withHead=True, nbClasses=nb_classes,
                                                  input_axes_order=axes_order)
        elif args.model == 'cnn2-sparseish':
            base_model = base_models_mtl.convNet2Sparseish(
                input_shape, withHead=True, nbClasses=nb_classes)
        elif args.model == 'cnn-sparse':
            base_model = base_models_mtl.cnnSparse(
                inputShape=input_shape, withHead=True, nbClasses=nb_classes)
        elif args.model == 'tripathi':
            # base_model = convNetDEAPTripathi(
            base_model = base_models_mtl.convNetDEAPTripathiReluSingleChannel(
                input_shape=input_shape,
                num_classes=num_classes,
                label_names=label_names,
                generate_head=True
            )
        elif args.model == 'mlp':
            base_model = base_models_mtl.mlp(
                input_shape=input_shape,
                num_classes=num_classes,
                label_names=label_names,
                generate_head=True
            )
        else:
            raise ValueError("Unsupported base model!")
        return base_model

    if args.model == 'cnn2':
        if not args.cnn2dense:
            base_model = base_models_mtl.convNet2(
                inputShape=input_shape,
                axes_order=axes_order,
                neuronsMLP=[])
        else:
            base_model = base_models_mtl.convNet2(
                inputShape=input_shape, axes_order=axes_order)
    elif args.model == 'cnn-sparse':
        base_model = base_models_mtl.cnnSparse(inputShape=input_shape)
    elif args.model == 'cnn2-sparseish':
        base_model = base_models_mtl.convNet2Sparseish(input_shape)
    elif args.model == 'normConv3':
        base_model = base_models_mtl.normConv3(
            input_shape=input_shape,
            units_mlp=[])
    else:
        raise ValueError("Unsupported base model!")

    if args.head_layout == 'dense':
        heads = generate_mtl_head(task_labels=label_names,
                                  neurons_per_head=args.neurons_per_head,
                                  layers_per_head=args.layers_per_head,
                                  number_classes=nb_classes,
                                  dense_dropout=args.head_dropout,
                                  input_model=base_model)
    elif args.head_layout == 'sparse':
        heads = generate_sparse_mtl_head(task_labels=label_names,
                                         number_classes=nb_classes,
                                         layers_per_head=args.layers_per_head,
                                         sizes_per_head=args.sizes_per_head,
                                         filters_per_head=args.filters_per_head,
                                         input_model=base_model)
    else:
        raise ValueError("Unsupported head layout!")

    mtl_model = tf.keras.Model(inputs=base_model.input, outputs=heads)

    return mtl_model


def train_mtl_model(model,
                    label_names,
                    args,
                    train_data,
                    input_shape,
                    val_data,
                    outpath,
                    deap_config=None):

    start = time.time()

    model.summary()
    model_plot = path.join(outpath, f"{args.model}_{args.tag}.png")
    tf.keras.utils.plot_model(model, to_file=model_plot,
                              show_shapes=True, show_layer_names=True,
                              dpi=320)
    print(f"Saved model img to {model_plot}")
    model_json = model.to_json(indent=2)
    jsonname = os.path.join(outpath, f"model{args.model}_{args.tag}.json")
    with open(jsonname, "w") as json_file:
        json_file.write(model_json)

    if args.dry_run:
        print('Dry-running, exiting.')
        return

    # callbacks
    tbpath = path.join(outpath, "tensorboard")
    symtbpath = path.join(args.output, "tensorboard", args.tag)
    if not os.path.exists(tbpath):
        os.makedirs(tbpath)
    if not os.path.exists(symtbpath):
        os.symlink(tbpath, symtbpath)
        print(f"Symlinked {tbpath} -> {symtbpath}")
    log_files_list = os.listdir(tbpath)
    if log_files_list != []:
        for fn in log_files_list:
            print(f"Deleting {path.join(tbpath, fn)}")
            shutil.rmtree(path.join(tbpath, fn))
        # os.makedirs(tbpath)
    checkpath = path.join(outpath, 'checkpoint/')
    if not os.path.exists(checkpath):
        os.makedirs(checkpath)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tbpath,
                                                 update_freq='epoch',
                                                 profile_batch=0,
                                                 # histogram_freq=5,
                                                 write_graph=True,
                                                 write_images=True)

    if args.dataset == "deap":
        test_data, val_data = val_data
        available = deap_config["files"]["available"]
        train_available = available["train"]
        validation_available = available["test"]

        train_batches = int(train_available / args.batch)
        validation_batches = int(validation_available / args.batch)

        print(
            f"DEAP: training on {train_available} samples, {train_batches} batches")
        print(
            f"      testing on {validation_available} samples, {validation_batches} batches")

        args.steps = train_batches
        funname = 'accuracy'

        monitorname = f"val_{label_names[0]}_out_accuracy"
        if len(label_names) == 1:
            monitorname = 'accuracy'

    elif args.dataset == "opportunity":
        test_data = None
        validation_batches = None
        monitorname = f"{label_names[0]}_out_fmeasure"
        if len(label_names) == 1:
            monitorname = 'fmeasure'

    evaluator = Evaluator(label_names)
    eval_dir = path.join(outpath, 'evaluation')
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)
    eval_callback = EvaluationCallback(
        val_data, label_names, num_classes, eval_dir)

    profile_callback = ProfileCallback()

    check_name = path.join(checkpath, f'{args.model}_{args.tag}.hdf5')

    check_callback = tf.keras.callbacks.ModelCheckpoint(check_name,
                                                        monitor=monitorname,
                                                        save_best_only=True,
                                                        mode='max',
                                                        # period=1,
                                                        save_freq='epoch',
                                                        save_weights_only=False)
    print(f"Checkpoint callback is monitoring the {monitorname} metric")

    optimizer = shared.build_optimizer(args.optimizer_args)

    print('Initiating the training phase ...')
    print("Hyperparameter summary:")
    optimizer_string = "default Adagrad(1.0)" if args.optimizer_args is None \
        else f"{args.optimizer_args['name']} with kwargs {jsons.dumps(args.optimizer_args['kwargs'])}"
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Tasks:        {', '.join(label_names)}")
    print(
        f"  Loss weights: [{', '.join(str(lw) for lw in args.loss_weights)}]")
    print(
        f"  Optimizer:      {optimizer_string}")

    metrics = shared.generate_metrics_dict(
        label_names, num_classes, args, dataset=args.dataset)
    losses = generate_loss_dict(label_names, args)
    class_weight = generate_class_weights(
        label_names,
        num_classes,
        null_weight=args.null_weight,
        non_null_weight=args.non_null_weight)

    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=args.loss_weights,
                  metrics=metrics,
                  # metrics=['acc', fmeasure]
                  )

    if args.dataset == "deap":
        callbacks = [
            tb_callback,
            # check_callback
        ]
    elif args.dataset == 'opportunity':
        callbacks = [
            tb_callback,
            check_callback,
            profile_callback,
            eval_callback
        ]

    hist = model.fit(
        train_data,
        verbose=1,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=test_data,
        validation_steps=validation_batches,
        # class_weight=class_weight,
        callbacks=callbacks)
    history = hist.history
    history_file = os.path.join(outpath, "history.json")
    with open(history_file, "w") as hf:
        j = jsons.dumps(history, indent=2)
        hf.write(j)
        print(f"Wrote {history_file}")

    # Save the weights of the network
    model_json = model.to_json()
    jsonname = os.path.join(outpath, f"model{args.model}_{args.tag}.json")
    with open(jsonname, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    hdfname = os.path.join(outpath, f"model{args.model}_{args.tag}_weights.h5")
    model.save_weights(hdfname)
    print("Saved model to folder:" + outpath)

    final_name = path.join(
        outpath, f'{args.model}_{args.tag}_model+weights.hdf5')
    model.save(final_name)

    print('##############################################')
    end = time.time()
    print('Total time used: %.2f seconds' % (end-start))
    print(f'Tensorboard log file generated in the directory {tbpath}')
    print('Use the command')
    print(f'    tensorboard --logdir {tbpath}')
    print('to read it')
    print()
    print('##############################################')


if __name__ == "__main__":

    # parser = utils.app.build_parser()
    args = parse_args(model_choices)

    if args.dataset is not None:
        args.output = args.output.replace("$dataset$", args.dataset.upper())
    else:
        raise ValueError()

    outpath = os.path.join(args.output, args.model, args.tag)
    logfile = os.path.join(outpath, f"{args.model}_{args.tag}.log")

    if not path.isdir(outpath):
        os.makedirs(outpath)

    with open(logfile, 'w') as log:
        sys.stdout = Unbuffered(sys.stdout, log)
        print(f"Logging to {logfile}")

        print_arguments(args)

        if not path.exists(outpath):
            os.makedirs(outpath)

        if args.dataset == 'deap':
            deap_folder = args.deap_path
            label_names, num_classes, all_names = select_channels_deap(
                args.labels)
            #deap_shape = args.deap_shape
            train_data, val_data, input_shape, deap_config = load_deap_data(
                args, selected_names=label_names)

        elif args.dataset == 'opportunity':
            nbSensors = args.opportunity_num_sensors
            if nbSensors == 107:
                pathExtension = path.join(
                    'all_sensors', str(args.opportunity_time_window))
            else:
                pathExtension = f'{nbSensors}_highest_var_sensors'
            channels = args.labels

            dataPath = path.join(args.opportunity_path, pathExtension)

            train_data, val_data, input_shape, label_names, num_classes = load_opportunity_data_mtl_tf(
                dataPath, args)
            deap_config = None
        else:
            raise ValueError

        mtl_model = build_mtl_model(
            args, input_shape, label_names, num_classes)

        train_mtl_model(model=mtl_model,
                        label_names=label_names,
                        args=args,
                        train_data=train_data,
                        val_data=val_data,
                        input_shape=input_shape,
                        outpath=outpath,
                        deap_config=deap_config)
