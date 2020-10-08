import argparse


def feature_args(parser):
    parser.add_argument('--sample-rate', default=16000, type=int)
    parser.add_argument('--win-length', default=.025, type=float)
    parser.add_argument('--hop-length', default=.01, type=float)
    parser.add_argument('--n-frames', default=200, type=int)
    parser.add_argument('--n-fft', default=512, type=int)
    parser.add_argument('--n-filterbanks', default=40, type=int)
    parser.add_argument(
        '--feat-type',
        default='spect',
        type=str,
        help='[mel | spect]'
    )
    parser.add_argument('--musan-path', default='/data/musan', type=str)
    parser.add_argument(
        '--rir-path',
        default='/data/RIRS_NOISES/simulated_rirs',
        type=str
    )
    parser.add_argument('--augment-prob', default=.8, type=float)
    return parser


def dataset_args(parser):
    parser.add_argument(
        '--dev-csv',
        default='/media/aj/wav_train_list.txt',
        type=str,
        help='training csv path'
    )
    parser.add_argument(
        '--eval-csv',
        default='/media/aj/wav_test_list.txt',
        type=str,
        help='testing csv path'
    )
    parser.add_argument(
        '--samples-per-speaker',
        default=2,
        type=int,
        help='num sample from each speaker in a minibatch'
    )
    return parser


def model_args(parser):
    parser.add_argument(
        '--trunk-net',
        default='resnet',
        type=str,
        help='trunk network type: [resnet | resnetse]'
    )
    parser.add_argument(
        '--pooling-net',
        default='sap',
        type=str,
        help='pooling network type: [tap | sap]'
    )
    parser.add_argument('--repr-dim', default=512, type=int)
    return parser


def training_hyper_params(parser):
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--num-epochs', default=10000, type=int)
    parser.add_argument('--update-interleaf', default=4, type=int)
    return parser


def criterion_args(parser):
    parser.add_argument(
        '--criterion-type',
        default='classification',
        type=str,
        help=(
            'specifies dataloader type according to loss type: [ cassification'
            ' | metriclearning]'
        )
    )
    parser.add_argument(
        '--criterion',
        default='cosface',
        type=str,
        help='criterion type: [cosface | psge2e |  protypical]'
    )
    parser.add_argument(
        '--criterion-lr',
        default=.001,
        type=float,
        help=(
            'learning rate of criterion (effective if it has some learnable '
            'parameter)'
        )
    )
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
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--criterion-path', default='', type=str)
    parser.add_argument('--optimizer-path', default='', type=str)
    parser.add_argument('--scheduler-path', default='', type=str)
    return parser


def scheduler(parser):
    parser.add_argument('--scheduler-step-size', default=1600, type=int)
    parser.add_argument('--scheduler-gamma', default=.9, type=float)
    return parser


def other_args(parser):
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
    parser.add_argument('--save-path', default='checkpoints')
    return parser


def create_argparser():
    parser = argparse.ArgumentParser(description='Training options')
    parsers_functions = [
        feature_args,
        dataset_args,
        model_args,
        training_hyper_params,
        criterion_args,
        resnet_args,
        cosface_args,
        psge2e_args,
        prototypical_args,
        load_model_args,
        scheduler,
        other_args,
    ]
    for parsers_function in parsers_functions:
        parser = parsers_function(parser)
    return parser
