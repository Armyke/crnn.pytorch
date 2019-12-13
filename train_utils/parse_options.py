import argparse

import torch


def parse_flags_and_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainRoot', required=True,
                        help='path to dataset')
    parser.add_argument('--valRoot', required=True,
                        help='path to dataset')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64,
                        help='input batch size')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256,
                        help='size of the lstm hidden state')
    parser.add_argument('--drop_out', default=0.1, type=float,
                        help='Drop out rate')
    parser.add_argument('--nepoch', type=int, default=25,
                        help='number of epochs to train for')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--cuda', action='store_true',
                        help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--pretrained', default='',
                        help="path to pretrained model (to continue training)")
    parser.add_argument('--alphabet', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--expr_dir', default='expr',
                        help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=500,
                        help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10,
                        help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500,
                        help='Interval to be displayed')
    # parser.add_argument('--saveInterval', type=int, default=500, help='Interval in which to save checkpoints')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for Critic, not used by adadelta')
    parser.add_argument('--lr_scheduler', action='store_true',
                        help='Whether to use a decaying lr_scheduler')
    parser.add_argument('--scheduler_type', type=str,
                        default='cosine',
                        help='If using  a lr_scheduler is possible to choose which:'
                             'Available types are:'
                             '- cosine (Default)')
    parser.add_argument('--decay', type=float, default=1,
                        help='Epoch decay factor, default 1 (no decay).'
                             'E.g. --decay 1.1 means that every epoch starting lr'
                             'will be divided by 1.1')
    parser.add_argument('--ad_lr', type=float, default=1,
                        help='learning rate for adadelta')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true',
                        help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true',
                        help='whether to keep ratio for image resize')
    parser.add_argument('--manualSeed', type=int, default=1234,
                        help='reproduce experiment')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--case', action='store_true',
                        help='Whether the alphabet is case sensitive (enabled), default in insensitive')
    parser.add_argument('--logdir',
                        help='Directory where to save TensorBoard logs, can be either local'
                             'or GCP bucket. If a bucket is needed to install TensorFlow')
    opt = parser.parse_args()

    # initialize dummy tensors to
    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    dummy_input_tensors = (image, text, length)

    return opt, dummy_input_tensors
