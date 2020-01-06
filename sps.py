from utils.app_sps import parse_args
import sys
import os
import tensorflow as tf
from utils.misc import Unbuffered, print_arguments, invoke_with_log
from models.base_models_mtl import convNet2SPS, convNet2CrossStitch, norm_conv3_crossstitch, norm_conv3_sps
import shared
from evaluation import Evaluator, EvaluationCallback
import shutil
from models.tensor_trace_norm import TensorTraceNorm
import time
from utils.deap import load_deap_data, select_channels_deap
import json


class SPS:

    def __init__(self, args, outpath):
        self.args = args
        self.data = args.dataset
        self.data_path = self._get_opportunity_data_path()
        self.out_path = outpath

    def _get_opportunity_data_path(self):
        if self.data == 'opportunity':
            nbSensors = self.args.opportunity_num_sensors
            if nbSensors == 107:
                pathExtension = os.path.join(
                    'all_sensors', str(self.args.opportunity_time_window))
            else:
                pathExtension = f'{nbSensors}_highest_var_sensors'
            channels = self.args.labels
            dataPath = os.path.join(self.args.opportunity_path, pathExtension)
            return dataPath
        elif self.data == 'deap':
            return args.deap_path

    def build_model_sps(self):
        print(
            f"Building model of type {self.args.model} for {len(self.label_names)} tasks")
        if self.args.model == "sps-opportunity":
            model, ttn_layers = convNet2SPS(
                input_shape=self.input_shape,
                label_names=self.label_names,
                nbClasses=self.num_classes,
                args=self.args)
            self.model = model
            self.ttn_layers = ttn_layers
        elif self.args.model == 'sps-deap':
            model, ttn_layers = norm_conv3_sps(
                input_shape=self.input_shape,
                label_names=self.label_names,
                nb_classes=self.num_classes
            )
            self.model = model
            self.ttn_layers = ttn_layers
        elif self.args.model == 'cross-stitch':
            if self.args.csmodel == 'normConv3':
                self.model = norm_conv3_crossstitch(
                    input_shape=self.input_shape,
                    nb_classes=self.num_classes,
                    label_names=self.label_names
                )
            else:
                self.model = convNet2CrossStitch(
                    input_shape=self.input_shape,
                    label_names=self.label_names,
                    nbClasses=self.num_classes,
                    args=self.args)
        else:
            raise AttributeError()

        self.model.summary()

        jsonname = os.path.join(
            self.out_path, f"{self.args.model}_{self.args.tag}.json")
        with open(jsonname, "w") as jf:
            jf.write(self.model.to_json(indent=2))

        model_plot = os.path.join(
            self.out_path, f"{self.args.model}_{self.args.tag}.png")
        tf.keras.utils.plot_model(self.model, to_file=model_plot,
                                  show_shapes=True, show_layer_names=True,
                                  dpi=320)
        print(f"Saved model img to {model_plot} and json to {jsonname}")

    def set_data(self, train_data, test_data, val_data,
                 input_shape, label_names, num_classes, deap_config):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.label_names = label_names
        self.num_classes = num_classes
        self.deap_config = deap_config

        if self.data == 'deap':
            deap_available = self.deap_config["files"]["available"]
            train_available = deap_available["train"]
            validation_available = deap_available["test"]

            train_batches = int(train_available / args.batch)
            validation_batches = int(validation_available / args.batch)
            print(
                f"DEAP: training on {train_available} samples, {train_batches} batches")
            print(
                f"      testing on {validation_available} samples, {validation_batches} batches")
            self.args.steps = train_batches
            self.args.validation_steps = validation_batches
        elif self.data == 'opportunity':
            self.args.validation_steps = None

    def generate_callbacks(self):
        callbacks = []

        tbpath = os.path.join(self.out_path, "tensorboard")
        symtbpath = os.path.join(args.output, "tensorboard", args.tag)
        if not os.path.exists(tbpath):
            os.makedirs(tbpath)
        if not os.path.exists(symtbpath):
            os.symlink(tbpath, symtbpath)
            print(f"Symlinked {tbpath} -> {symtbpath}")
        log_files_list = os.listdir(tbpath)
        if log_files_list != []:
            for fn in log_files_list:
                print(f"Deleting {os.path.join(tbpath, fn)}")
                shutil.rmtree(os.path.join(tbpath, fn))
        checkpath = os.path.join(self.out_path, 'checkpoint/')
        if not os.path.exists(checkpath):
            os.makedirs(checkpath)

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tbpath,
                                                     update_freq='epoch',
                                                     write_graph=True,
                                                     write_images=True)
        callbacks.append(tb_callback)

        check_name = os.path.join(checkpath, f'{args.model}_{args.tag}.hdf5')
        if self.data == 'opportunity':
            monitorname = f"out_{self.label_names[0]}_fmeasure"
            if len(self.label_names) == 1:
                monitorname = 'fmeasure'
        elif self.data == 'deap':
            monitorname = f"val_out_{self.label_names[0]}_accuracy"
        check_callback = tf.keras.callbacks.\
            ModelCheckpoint(check_name,
                            monitor=monitorname,
                            save_best_only=True,
                            mode='max',
                            save_freq='epoch',
                            save_weights_only=False)
        callbacks.append(check_callback)

        if self.data == 'opportunity':
            evaluator = Evaluator(self.label_names)
            eval_dir = os.path.join(outpath, 'evaluation')
            if not os.path.isdir(eval_dir):
                os.makedirs(eval_dir)
            eval_callback = EvaluationCallback(
                self.val_data,
                self.label_names,
                self.num_classes,
                eval_dir)
            callbacks.append(eval_callback)
        return callbacks

    def loss_trace_norm(self, ttn_layers, base_fun, ttn_weight=None):

        def loss(y_true, y_pred):
            def _ttn(layer_idx, ttn_layer):
                weights = [self.model.get_layer(
                    name=conv_name).kernel for conv_name in ttn_layer]
                shapeX = weights[0].get_shape().as_list()
                dimX = len(shapeX)
                # weights = [l.weights[0] for l in layers]
                stack = tf.stack(weights, axis=dimX)
                t = TensorTraceNorm(stack)
                tr = tf.reduce_sum(t)
                return tr
            cce = base_fun(y_true, y_pred)
            ttn = [_ttn(layer_idx, ttn_layer)
                   for layer_idx, ttn_layer
                   in enumerate(ttn_layers)]

            sttn = tf.reduce_sum(ttn)
            if ttn_weight is not None:
                sttn = tf.math.scalar_mul(ttn_weight,
                                          sttn)
            final_loss = cce + sttn
            return final_loss

        if ttn_layers is None:
            return base_fun

        return loss

    def get_loss_fun(self):
        if "sps" in self.args.model:
            if self.data == 'deap':
                if self.args.deap_one_hot:
                    base_fun = tf.keras.losses.categorical_crossentropy
                else:
                    base_fun = tf.keras.losses.sparse_categorical_crossentropy
            else:
                base_fun = tf.keras.losses.categorical_crossentropy
            print(
                f"loss will be {base_fun.__name__} with a trace loss weight of {args.traceloss}")
            return self.loss_trace_norm(self.ttn_layers,
                                        base_fun,
                                        self.args.traceloss)
        elif self.args.model == "cross-stitch":
            # return tf.keras.losses.categorical_crossentropy
            if self.args.deap_one_hot:
                return tf.keras.losses.categorical_crossentropy
            else:
                return tf.keras.losses.sparse_categorical_crossentropy
        else:
            raise ValueError()

    def train(self):
        self.callbacks = self.generate_callbacks()
        self.loss_fun = self.get_loss_fun()
        self.optimizer = shared.build_optimizer(self.args.optimizer_args)
        optimizer_string = "default Adagrad(1.0)" if args.optimizer_args is None else f"{args.optimizer_args['name']} with kwargs {json.dumps(args.optimizer_args['kwargs'])}"
        self.metrics = shared.generate_metrics_dict(
            self.label_names,
            self.num_classes,
            self.args,
            self.data,
            out_format_string="out_{ln}"
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fun,
            loss_weights=self.args.loss_weights,
            metrics=self.metrics
        )

        print('Initiating the training phase ...')
        print("Hyperparameter summary:")
        print(f"  Epochs:       {self.args.epochs}")
        print(f"  Batch size:   {self.args.batch}")
        if self.data == 'opportunity':
            print(f"  Window size:  {self.args.opportunity_time_window}")
            print(f"  Sensors:      {self.args.opportunity_num_sensors}")
        print(f"  Tasks:        {', '.join(self.label_names)}")
        print(f"  Loss weights: \
            [{', '.join(str(lw) for lw in self.args.loss_weights)}]")
        print(f"  Optimizer:    {optimizer_string}")

        start = time.time()
        print(f"\n  Current Unix time: {start}")

        self.model.fit(
            self.train_data,
            verbose=1,
            epochs=self.args.epochs,
            steps_per_epoch=self.args.steps,
            validation_data=self.test_data,
            validation_steps=self.args.validation_steps,
            callbacks=self.callbacks)

        print('##############################################')
        end = time.time()
        print('Total time used: %.2f seconds' % (end-start))

        # Save the weights of the network
        model_json = self.model.to_json()
        jsonname = os.path.join(
            self.out_path,
            f"model{self.args.model}_{self.args.tag}.json")
        with open(jsonname, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        hdfname = os.path.join(
            self.out_path,
            f"model{self.args.model}_{self.args.tag}_weights.h5")
        self.model.save_weights(hdfname)
        print("Saved model to folder:" + self.out_path)

        final_name = os.path.join(
            outpath, f'{self.args.model}_{self.args.tag}_model+weights.hdf5')
        self.model.save(final_name)


def load_data(args, sps):
    if args.dataset == 'opportunity':
        train_data, val_data, input_shape, label_names, num_classes = shared.load_opportunity_data_mtl_tf(
            sps.data_path, args)
        sps.set_data(train_data, None, val_data,
                     input_shape, label_names, num_classes, None)
    elif args.dataset == 'deap':
        label_names, num_classes, all_names = select_channels_deap(
            args.labels)
        data_train, data_test, shape, config = load_deap_data(
            args,
            label_names)
        data_test_test, _ = data_test  # do not need the val dataset which is unbatches
        sps.set_data(data_train, data_test_test, None,
                     shape, label_names, num_classes, config)


def training_loop(args, outpath):
    print_arguments(args)
    sps = SPS(args, outpath)
    load_data(args, sps)
    sps.build_model_sps()

    if not args.dry_run:
        sps.train()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset is not None:
        args.output = args.output.replace("$dataset$", args.dataset.upper())
    else:
        raise ValueError()
    outpath = os.path.join(args.output, args.model, args.tag)
    logfile = os.path.join(outpath, f"{args.model}_{args.tag}.log")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    with open(logfile, "w") as log:
        sys.stdout = Unbuffered(sys.stdout, log)
        print(f"Logging to {logfile}")
        training_loop(args=args, outpath=outpath)

        print("\nDone.", flush=True)
