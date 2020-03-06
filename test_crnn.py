import os
import json
from operator import itemgetter
from glob import glob
from argparse import ArgumentParser

from PIL import Image

from models.load_pretrained_crnn import load_model
from prediction_getter import predict_image


def test_on_list(path_list, model_path):

    model = load_model(model_path)

    predictions_list = []

    for path in path_list:

        # json_path = path.replace(path[-4:], "__metadata.json")
        #
        # with open(json_path, 'r') as j:
        #     info_dict = json.load(j)
        #
        # label = info_dict['text']

        label = path.split(os.sep)[-1].split('_')[0].replace('-', '').replace('0', '')

        # run model on each image
        image = Image.open(path).convert('L')

        raw_pred, sim_pred = predict_image(image, model)

        print('%-20s => %-20s \t label: %-20s' % (raw_pred, sim_pred, label))

        predictions_list.append({'filename': path, 'label': label,
                                 'pred': sim_pred})

    return predictions_list


def main():
    parser = ArgumentParser()
    parser.add_argument('--test_images_dir', required=True)
    parser.add_argument('--model_path', default='./demo.pth')
    parser.add_argument('--compare_models', default=-1,
                        help='If enable expects in model_path a directory'
                             'where multiple models are stored for compare.'
                             'Print for each model his accuracy and'
                             ' selects the higher one')
    args = parser.parse_args()

    images_path_list = glob(os.path.join(args.test_images_dir, '*', '*', '*png'))
    images_path_list += glob(os.path.join(args.test_images_dir, '*', '*', '*jpg'))

    if args.compare_models == -1:
        pred_list = test_on_list(images_path_list, args.model_path)

        # print output result
        wrong_pred = 0
        for item in pred_list:
            if item['label'].lower() != item['pred']:
                wrong_pred += 1

        print('Word Accuracy {}'.format(1-float(wrong_pred)/len(images_path_list)))

    else:
        models_list = glob(os.path.join(args.model_path, '*.pth'))

        out_list = []
        for model in models_list:
            pred_list = test_on_list(images_path_list, model)

            # print output result
            wrong_pred = 0
            for item in pred_list:
                if item['label'].lower() != item['pred']:
                    wrong_pred += 1

            acc = 1 - float(wrong_pred) / len(images_path_list)

            print('Word Accuracy {}'.format(acc))

            out_list.append([model, acc])

        out_list.sort(key=itemgetter(1), reverse=True)

        print("Best model is {} with an accuracy of {}".format(out_list[0][0], out_list[0][1]))

        print("Whole scores:")
        print(out_list)


if __name__ == '__main__':
    main()
