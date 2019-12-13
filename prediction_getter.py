import torch
from torch.autograd import Variable

import dataset
import utils


alphabet = 'abcdefghijklmnopqrstuvwxyz'


def predict_image(image, model):

    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))

    image = transformer(image)

    image = image.view(1, *image.size())
    image = Variable(image)

    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return raw_pred, sim_pred
