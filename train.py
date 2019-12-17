from __future__ import print_function
from __future__ import division

import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.utils.data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from torch.nn import CTCLoss

import numpy as np

# from warpctc_pytorch import CTCLoss

import utils
import dataset

from models.crnn import CRNN
from train_utils.train_eval_fns import validate_model, train_batch
from train_utils.parse_options import parse_flags_and_input
from train_utils.custom_lr_schedulers import adjust_learning_rate


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main(opt, dummy_input_tensors_list):

    img, txt, batch_length = dummy_input_tensors_list

    # Set up TensorBoard writer
    if not opt.logdir:
        tb_dir = os.path.join(opt.expr_dir, 'tb_logs')
    else:
        tb_dir = os.path.join(opt.logdir, opt.expr_dir.split(os.sep)[-1] + '_tb_logs')

    logs_writer = SummaryWriter(tb_dir,
                                flush_secs=60)

    print(opt)

    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    assert train_dataset
    # if not opt.random_sample:
    #     sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
    # else:
    #     sampler = None

    sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                               shuffle=True, sampler=sampler,
                                               num_workers=int(opt.workers),
                                               collate_fn=dataset.alignCollate(imgH=opt.imgH,
                                                                               imgW=opt.imgW,
                                                                               keep_ratio=opt.keep_ratio))
    test_dataset = dataset.lmdbDataset(root=opt.valRoot,
                                       transform=dataset.resizeNormalize((100, 32)))

    nclass = len(opt.alphabet) + 1
    nc = 1

    case = False if opt.case else True

    converter = utils.strLabelConverter(opt.alphabet, ignore_case=case)
    loss_fn = CTCLoss()

    crnn = CRNN(opt.imgH, nc, nclass, opt.nh, drop_out=opt.drop_out)
    crnn.apply(weights_init)

    fine_tune = False
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        fine_tune = True
        if opt.cuda:

            from collections import OrderedDict

            state_dict = torch.load(opt.pretrained)
            new_state_dict = OrderedDict()
            for key, layer in state_dict.items():
                new_key = key[7:]
                new_state_dict[new_key] = layer
            crnn.load_state_dict(new_state_dict)

        else:
            crnn.load_state_dict(torch.load(opt.pretrained))

    print(crnn)

    if opt.cuda:
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        img = img.cuda()
        loss_fn = loss_fn.cuda()

    img = Variable(img)
    txt = Variable(txt)
    b_length = Variable(batch_length)

    inputs = (img, txt, b_length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                               betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=opt.ad_lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    # set up scheduler if specified
    if opt.lr_scheduler:
        if opt.scheduler_type == 'cosine':
            print('Using cosine with warm restarts learning rate scheduler...')
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                    int(len(train_loader) / 3),
                                                    2,
                                                    eta_min=1e-5)
        else:
            raise NotImplementedError("Available types for scheduler are:"
                                      "- cosine (Default)")

    else:
        scheduler = False

    if opt.decay > 1:
        print("Decaying lr_max each epoch of a {} factor".format(opt.decay))

    for epoch in range(opt.nepoch):
        train_iter = iter(train_loader)

        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            step = epoch * len(train_loader) + i

            batch_loss, batch_acc = train_batch(inputs, crnn, loss_fn,
                                                train_iter, optimizer, converter,
                                                tb_writer=logs_writer,
                                                step=step,
                                                scheduler=scheduler)

            loss_avg.add(batch_loss)

            i += 1

            if i % opt.valInterval == 0:
                val_loss, val_acc = validate_model(opt, inputs, crnn,
                                                   test_dataset, loss_fn,
                                                   converter,
                                                   tb_writer=logs_writer,
                                                   step=step)

                if fine_tune:
                    ckpt_path = '{}/netCRNN_fine_tune_{}_{}_{:.4f}_{:.4f}_{:.4f}.pth'.format(opt.expr_dir,
                                                                                             epoch, i,
                                                                                             loss_avg.val(),
                                                                                             val_loss,
                                                                                             val_acc)
                else:
                    ckpt_path = '{}/netCRNN_{}_{}_{:.4f}_{:.4f}_{:.4f}.pth'.format(opt.expr_dir, epoch,
                                                                                   i, loss_avg.val(),
                                                                                   val_loss,
                                                                                   val_acc)

                torch.save(
                    crnn.state_dict(), ckpt_path)

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f\tAcc: %f' %
                      (epoch, opt.nepoch, i, len(train_loader), loss_avg.val(), batch_acc))
                loss_avg.reset()

                # do checkpointing
                # if i % opt.saveInterval == 0:
                #     torch.save(
                #         crnn.state_dict(),
                #         '{0}/netCRNN_{1}_{2}_{3}_{4}.pth'.format(opt.expr_dir, epoch, i, loss, acc))

        if opt.decay > 1:
            # update optimizer learning rate
            new_lr = optimizer.param_groups[0]['initial_lr'] / opt.decay
            adjust_learning_rate(optimizer, new_lr)

            if scheduler:
                scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                        int(len(train_loader) / 3),
                                                        2,
                                                        eta_min=1e-5)

        elif opt.decay != 1:
            print("Wrong value encountered for decay:"
                  " must be a float greater than 1. "
                  "Skipping learning rate decay...")

    logs_writer.close()


if __name__ == '__main__':
    OPT, DUMMY_INPUT = parse_flags_and_input()
    main(OPT, DUMMY_INPUT)
