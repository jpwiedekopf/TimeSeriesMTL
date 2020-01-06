import argparse
import json


def intlist(s): return [int(item) for item in s.strip().split(',')]


def floatlist(s): return [float(item) for item in s.strip().split(',')]


def js(s): return json.loads(s)


def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=str)
    parser.add_argument("dataset", type=str.lower,
                        choices=['deap', 'opportunity'])
    parser.add_argument('labels', type=intlist)

    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument("-o", "--output", type=str,
                        default='/models/$dataset$/MTL-SPS/')
    parser.add_argument('--shuffle-buffer', type=int, default=1000)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-s", "--steps", type=int,
                        default=1000, help='Steps per epoch')
    parser.add_argument("-b", "--batch", type=int, default=500)
    parser.add_argument("--rate", type=float, default=0.05),
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument(
        '--loss-weights', type=floatlist, default=[1, 1])
    parser.add_argument("--optimizer-args", type=js, default=None)

    parser.add_argument("--opportunity-path", type=str,
                        default=r'/data/opportunity/window_labels-mtl-tf')
    parser.add_argument("--opportunity-num-sensors", type=int, default=107,
                        choices=[5, 10, 20, 50, 80, 107])
    parser.add_argument("--opportunity-time-window", type=int,
                        default=64, choices=[32, 64, 96])

    parser.add_argument("--deap-path", type=str,
                        default=r'/data/deap/windowed')
    parser.add_argument("--deap-no-one-hot",
                        action='store_false', dest='deap_one_hot')
    parser.set_defaults(deap_one_hot=True)

    sp = parser.add_subparsers(dest="model")
    sp.required = True
    sps_opp_parser = sp.add_parser("sps-opportunity")
    add_cnn2_arguments(sps_opp_parser, required=True)
    sps_opp_parser.add_argument('--traceloss', type=float, default=0.001,
                                help='Scalar weighing the tensor trace component of the loss')

    sps_deap_parser = sp.add_parser("sps-deap")
    sps_deap_parser.add_argument('--traceloss', type=float, default=0.001,
                                 help='Scalar weighing the tensor trace component of the loss')

    cross_parser = sp.add_parser("cross-stitch")
    cross_parser.add_argument("csmodel", choices=['normConv3', 'cnn2'])
    add_cnn2_arguments(cross_parser)

    return parser


def add_cnn2_arguments(subparser, required=False):
    subparser.add_argument("--numlayers", type=int, default=3)
    subparser.add_argument("--numfilters", type=intlist, required=required)
    subparser.add_argument("--numkerns", type=intlist, required=required)
    subparser.add_argument("--poolsizes", type=intlist, required=required)
    subparser.add_argument("--numdenselayers", type=int, default=2)
    subparser.add_argument("--numdenseunits", type=intlist, required=required)


def parse_args():
    parser = make_parser()
    args = parser.parse_args()
    if args.model == "cnn2":
        if args.numlayers != len(args.numfilters) or \
                args.numlayers != len(args.numkerns) or \
                args.numlayers != len(args.poolsizes) or \
                args.numdenselayers != len(args.numdenseunits):
            raise AttributeError("Check argument lengths!")
    return args
