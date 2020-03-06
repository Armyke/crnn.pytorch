import torch
from torch.autograd import Variable

from cv2 import putText, FONT_ITALIC, LINE_AA
from torch import Tensor, cat

import utils

from dataset import alignCollate


def train_batch(inputs, net, loss_fn, dataloader, optimizer, converter,
                tb_writer=None, scheduler=False, step=None):
    """

    :param inputs: List of the input tensors of the model
    :param net: Model pytorch object
    :param loss_fn: Loss function of the model
    :param dataloader: Dataloader pytorch object (iterable)
    :param optimizer: Optimizer of the model (e. g. torch.nn.optim.Adam)
    :param converter:
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param scheduler: Eventual Learning rate scheduler
    :param step: Counter for TB logging

    :return: value of loss and accuracy calculated over the batch
    """

    if tb_writer and step is None:
        raise ValueError("If using a tb_writer, "
                         "step counter must be defined")

    image, text, length = inputs

    data = dataloader.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.load_data(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.load_data(text, t)
    utils.load_data(length, l)

    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    loss = loss_fn(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = preds.max(2)

    if len(preds.shape) == 3:
        preds = preds.squeeze(2)

    n_correct = 0
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    for pred, target in zip(sim_preds, cpu_texts):
        if pred == target.lower():
            n_correct += 1

    accuracy = n_correct / float(batch_size)

    if scheduler:
        scheduler.step()

    if tb_writer:
        tb_writer.add_scalar('loss/train_loss', loss.data.item(),
                             step)
        tb_writer.add_scalar('metrics/train_acc', accuracy,
                             step)

        sample = ((cpu_images[0] + 1) / 2 * 255).detach().numpy().astype('uint8')[0]

        drawn_sample = putText(sample,
                               sim_preds[0],
                               (0, 10),
                               FONT_ITALIC,
                               0.5,
                               (0, 255, 0),
                               2,
                               LINE_AA,
                               bottomLeftOrigin=False)

        tb_writer.add_images('training sample',
                             drawn_sample,
                             step, dataformats='HW')

        if scheduler:
            tb_writer.add_scalar('learning rate',
                                 scheduler.get_lr()[-1],
                                 step)

    return loss, accuracy


def validate_model(opt, inputs, net, dataset, loss_fn, converter,
                   max_iter=500, tb_writer=None, step=None):
    """

    :param opt: flags argument parser
    :param inputs: List of the input tensors of the model
    :param net: Model pytorch object
    :param dataset:
    :param loss_fn: Loss function of the model
    :param converter:
    :param max_iter:
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param step: Counter for TB logging

    :return: val_loss, val_acc: Loss and accuracy scalars
    """

    image, text, length = inputs

    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                              batch_size=opt.batchSize,
                                              num_workers=int(opt.workers),
                                              collate_fn=alignCollate(imgH=opt.imgH,
                                                                      imgW=opt.imgW,
                                                                      keep_ratio=opt.keep_ratio))
    val_iter = iter(data_loader)

    n_correct = 0
    count = 0
    loss_avg = utils.Averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.load_data(text, t)
        utils.load_data(length, l)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = loss_fn(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)

        if len(preds.shape) == 3:
            preds = preds.squeeze(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            count += 1
            if pred == target.lower():
                n_correct += 1

    # TODO fix bug when batch size is 1
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]

    to_log_samples = []

    for idx, (raw_pred, pred, gt) in enumerate(zip(raw_preds, sim_preds, cpu_texts)):

        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        if tb_writer:

            sample = ((cpu_images[idx] + 1) / 2 * 255).detach().numpy().astype('uint8')[0]

            drawn_sample = putText(sample,
                                   pred,
                                   (0, 10),
                                   FONT_ITALIC,
                                   0.5,
                                   (0, 255, 0),
                                   2,
                                   LINE_AA,
                                   bottomLeftOrigin=False)

            to_log_samples.append(Tensor(drawn_sample).unsqueeze(0))

    tb_writer.add_images('validation samples',
                         cat(to_log_samples, 1) / 255,
                         step,
                         dataformats='CHW')

    accuracy = n_correct / float(count)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))

    if tb_writer:
        tb_writer.add_scalar('loss/val_loss', loss_avg.val(),
                             step)
        tb_writer.add_scalar('metrics/val_acc', accuracy,
                             step)

    return loss_avg.val(), accuracy
