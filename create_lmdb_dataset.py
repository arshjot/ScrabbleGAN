"""a modified version of https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/create_lmdb_dataset.py"""

import os
import io
import lmdb
import argparse
import pickle as pkl
from PIL import Image
from utils.data_utils import WordMap
from config import Config


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)


def createDataset(config, output_path, generate_additional=0, checkpt_path=None, char_map=None):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        config  : config file
        output_path : LMDB output path
        generate_additional     : number of additional images to be generated
        checkpt_path : Path of the model checkpoint file to be used for generating additional images
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(config.data_file, 'rb') as f:
        data = pkl.load(f)
    word_data = data['word_data']
    idx_to_id = {i: w_id for i, w_id in enumerate(word_data.keys())}
    word_map = WordMap(char_map)

    nSamples = len(word_data)
    for i in range(nSamples):
        w_id = idx_to_id[i]

        # Get image and label
        lab, img = word_data[w_id]
        img = Image.fromarray(img).convert("RGB")
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='tiff')
        wordBin = imgByteArr.getvalue()

        label = word_map.decode([lab])[0]

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        cache[imageKey] = wordBin
        cache[labelKey] = label.encode()

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
    config = Config
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_path", required=True, type=str,
                    help="Path of the output lmdb file")
    ap.add_argument("-cm", "--char_map_path", required=True, type=str,
                    help="Path of the character map pkl file")
    ap.add_argument("-c", "--checkpoint_path", required=False, type=str,
                    help="Path of the model checkpoint file to be used")
    ap.add_argument("-n", "--generate_additional", required=False, type=int,
                    help="number of additional images to be generated")
    args = vars(ap.parse_args())
    output_path = args['output_path']
    char_map_path = args['char_map_path']
    gen_add = args['generate_additional'] if args['generate_additional'] is not None else 0
    checkpoint_path = args['checkpoint_path'] if args['checkpoint_path'] is not None else None

    with open(char_map_path, 'rb') as f:
        char_map = pkl.load(f)
    createDataset(config, output_path, gen_add, checkpoint_path, char_map=char_map)
