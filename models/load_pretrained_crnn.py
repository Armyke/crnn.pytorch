import torch

from models import crnn as crnn


def load_model(model_path, nh, alphabet='abcdefghijklmnopqrstuvwxyz'):
    # initialize crnn model
    model = crnn.CRNN(32, 1, len(alphabet) + 1, nh)

    # load model
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    except Exception as e:

        from collections import OrderedDict

        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for key, layer in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = layer
        model.load_state_dict(new_state_dict)

    model.eval()

    return model
