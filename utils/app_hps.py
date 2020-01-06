import argparse
import json


def build_parser(model_choices):
    def intlist(s): return [int(item) for item in s.strip().split(',')]
    def floatlist(s): return [float(item) for item in s.strip().split(',')]
    def js(s): return json.loads(s)
    topparser = argparse.ArgumentParser()
    topparser.add_argument("model", choices=model_choices,
                           help="The model to use")
    topparser.add_argument("tag", type=str,
                           help="Tag which to append to all generated files to describe this training run")

    topparser.add_argument("dataset", type=str.lower,
                           choices=['deap', 'opportunity'])

    topparser.add_argument("--opportunity-path", type=str,
                           default=r'/data/opportunity/window_labels-mtl-tf')
    topparser.add_argument("--opportunity-num-sensors", type=int, default=107,
                           choices=[5, 10, 20, 50, 80, 107])
    topparser.add_argument("--opportunity-time-window", type=int,
                           default=64, choices=[32, 64, 96])

    topparser.add_argument(
        "--cnn2-no-dense", action="store_false", default=True, dest="cnn2dense")

    # topparser.add_argument("--deap-validation", type=intlist,
    #                        default=intlist("5"))
    topparser.add_argument("--deap-path", type=str,
                           default=r'/data/deap/windowed')
    topparser.add_argument("--deap-no-one-hot",
                           action='store_false', dest='deap_one_hot')
    topparser.set_defaults(deap_one_hot=True)

    topparser.add_argument(
        "--output", default='/models/$dataset$/MTL-HPS', type=str)

    topparser.add_argument('--labels',
                           type=intlist, help='The channel IDs for labelling, delimit with comma',
                           required=True)
    topparser.add_argument("--dry-run", help="If given, do not train,\
        just instantiate the model and save the image to\
        disk", action="store_true")

    topparser.add_argument('--shuffle-buffer', type=int, default=1000)
    topparser.add_argument("-e", "--epochs", type=int, default=50)
    topparser.add_argument("-s", "--steps", type=int,
                           default=1000, help='Steps per epoch')
    topparser.add_argument("-b", "--batch", type=int, default=500)
    topparser.add_argument("--rate", type=float, default=0.05),
    topparser.add_argument("--dropout", type=float, default=0.4)

    topparser.add_argument(
        '--loss-weights', type=floatlist, default=[1, 1])
    topparser.add_argument("--null-weight", type=float, default=1.0)
    topparser.add_argument("--non-null-weight", type=float, default=1.0)

    topparser.add_argument("--optimizer-args", type=js, default=None)

    subparsers = topparser.add_subparsers(title='MTL Head Layouts',
                                          dest='head_layout')
    subparsers.required = True

    subparsers.add_parser('none', )

    denseparser = subparsers.add_parser('dense', )
    denseparser.add_argument('--neurons-per-head',
                             type=intlist, required=True, action="append")
    denseparser.add_argument(
        '--layers-per-head', type=intlist, required=True)
    denseparser.add_argument(
        "--head-dropout", type=floatlist, required=True, action="append")

    sparseparser = subparsers.add_parser('sparse')
    sparseparser.add_argument(
        '--layers-per-head', type=intlist, required=True)
    sparseparser.add_argument(
        '--sizes-per-head', type=intlist, required=True)
    sparseparser.add_argument('--filters-per-head',
                              type=intlist, required=True)

    return topparser


def parse_args(model_choices):
    parser = build_parser(model_choices)
    return parser.parse_args()
