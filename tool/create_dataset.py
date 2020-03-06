from argparse import ArgumentParser
from glob import glob
from random import shuffle
import os
import lmdb
import cv2
import numpy as np
import json

from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(str(k).encode(), str(v))


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    PARSER = ArgumentParser()
    PARSER.add_argument('--images_dir', help='Path to the directory where images are',
                        required=True)
    PARSER.add_argument('--out_dir', help='Path to the output directory',
                        required=True)
    PARSER.add_argument('--from_json', default=True,
                        help='Whether to extract labels from json files or not')
    PARSER.add_argument('--split', default=0.8, type=float,
                        help='Train split percentage')
    PARSER.add_argument('--alphabet', default="abcdefghijklmnopqrstuvwxyz",
                        type=str,
                        help='Alphabet as string. Default value is:'
                             '- "abcdefghijklmnopqrstuvwxyz"')

    ARGS = PARSER.parse_args()

    if ARGS.from_json not in ('True', 'False'):
        raise ValueError("Expected boolean for from_json arg (True or False)")
    else:
        ARGS.from_json = True if ARGS.from_json == 'True' else False

    images_dir_path = ARGS.images_dir
    output_dir = ARGS.out_dir
    train_dir = os.path.join(output_dir, 'train')
    eval_dir = os.path.join(output_dir, 'eval')

    if not os.path.exists(output_dir):
        os.makedirs(train_dir)
        os.makedirs(eval_dir)
    elif not os.path.exists(train_dir) or not os.path.exists(eval_dir):
        os.mkdir(train_dir)
        os.mkdir(eval_dir)

    tmp_image_path_list = glob(os.path.join(images_dir_path, '*', '*', '*.png'))
    tmp_image_path_list += glob(os.path.join(images_dir_path, '*', '*', '*.jpg'))

    shuffle(tmp_image_path_list)

    tmp_labels_list = []

    if ARGS.from_json:

        print("Starting label extraction from metadata\n")

        for path in tqdm(tmp_image_path_list):

            json_path = path.replace(path[-4:], "__metadata.json")

            with open(json_path, 'r') as j:
                tmp_labels_list.append(json.load(j)['text'])
                j.close()
    else:

        print("Starting label extraction as first element of image name in snake case")
        for path in tqdm(tmp_image_path_list):

            img_name = path.split(os.sep)[-1]
            label = img_name.split('_')[0]

            if '-' in label:
                label = label.replace('-', '')
            if '0' in label:
                label = label.replace('0', '')
            if '1' in label:
                label = label.replace('0', '')

            tmp_labels_list.append(label)

    print("Filtering images which contains chars not in alphabet\n")

    tmp = []
    for img, label in tqdm(zip(tmp_image_path_list, tmp_labels_list)):
        is_in_alph = True

        try:
            encoding_check = label.encode()
            for char in list(label.lower()):
                if char not in ARGS.alphabet:
                    is_in_alph = False

            if is_in_alph:
                tmp.append((img, label.lower()))

        except KeyError("Non ascii character found in label, skipping image"):
            continue

    image_path_list, labels_list = zip(*tmp)

    split_percentage = ARGS.split

    if split_percentage:

        len_train = int(len(image_path_list) * split_percentage)

        print("Labelling finished, starting LMDB dataset creation\n")

        createDataset(train_dir, image_path_list[:len_train], labels_list[:len_train])
        createDataset(eval_dir, image_path_list[len_train:], labels_list[len_train:])

    else:
        print("Labelling finished, starting LMDB dataset creation\n")
        print("--split was either 0 or None, creating a single lmdb file in train dir.")
        createDataset(train_dir, image_path_list, labels_list)
