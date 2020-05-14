import argparse


def common_args(parser):
    # dataset
    parser.add_argument(
        '--dev-csv',
        default='/media/aj/wav_train_list.txt',
        type=str,
        help='training csv path'
    )
    parser.add_argument(
        '--test-csv',
        default='/media/aj/wav_test_list.txt',
        type=str,
        help='testing csv path'
    )
    parser.add_argument('--sample-rate', default=16000, type=int)
    parser.add_argument('--win-length', default=400, type=int)
    parser.add_argument('--hop-length', default=160, type=int)
    parser.add_argument('--num-frames', default=200, type=int)
    parser.add_argument(
        '--criterion-type',
        default='classification',
        type=str,
        help='specifies dataloader type according to loss type: cassification\
             | metriclearning'
    )
    parser.add_argument(
        '--spk-samples',
        default=2,
        type=int,
        help='num sample from each speaker in a minibatch'
    )
    parser.add_argument('--num-filterbanks', default=40, type=int)
    # model selection
    parser.add_argument(
        '--trunk-net',
        default='resnet',
        type=str,
        help='trunk network type: (currently only) resnet'
    )
    parser.add_argument(
        '--polling-net',
        default='tap',
        type=str,
        help='polling network type: tap | sap'
    )
    # training hyper parameters
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--num-epochs', default=10000, type=int)
    # criterion
    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument(
        '--criterion',
        default='cosface',
        type=str,
        help='criterion type: cosface | psge2e |  protypical'
    )
    parser.add_argument(
        '--criterion-lr',
        default=.001,
        type=float,
        help='learning rate of criterion (effective if it has some learnable\
             parameter)'
    )
    parser.add_argument(
        '--test-interleaf',
        default=10,
        type=int,
        help='every n epoch do a evaluation and report EER on tensorboard'
    )
    # log path
    parser.add_argument('--logdir', default='log', type=str)
    # saving results
    parser.add_argument('--save-checkpoint', action='store_true')
    return parser


def resnet_args(parser):
    # nerwork params
    parser.add_argument(
        '--layers',
        default=[3, 4, 6, 3],
        type=int,
        nargs='+'
    )
    return parser


def cosface_args(parser):
    parser.add_argument('--m', default=.2, type=float)
    parser.add_argument('--s', default=30, type=float)
    return parser


def psge2e_args(parser):
    parser.add_argument('--init-m', default=-5., type=float)
    parser.add_argument('--init-s', default=10., type=float)
    return parser


def prototypical_args(parser):
    return parser


def load_model_args(parser):
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--criterion-path', default='', type=str)
    parser.add_argument('--optimizer-path', default='', type=str)
    return parser


def add_args(parser):
    parsers_functions = [
        common_args,
        resnet_args,
        cosface_args,
        psge2e_args,
        prototypical_args,
        load_model_args
    ]
    for parsers_function in parsers_functions:
        parser = parsers_function(parser)
    return parser
