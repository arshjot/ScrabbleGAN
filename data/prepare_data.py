"""
This file is for preparing IAM/RIMES word level dataset
Please first download IAM word level dataset and extract it in a new folder here named 'IAM'
Ensure the following directory structure is followed:
├── data
|   ├── IAM
|       └──ascii
|           └──words.txt
|       └──words
|           └──a01
|           └──a02
|           .
|           .
|       └──original_partition
|           └──te.lst, tr.lst, va1.lst, va2.lst
|   ├── RIMES
|       └──ground_truth_training_icdar2011.txt
|       └──training
|           └──lot_1
|           └──lot_2
|           .
|           .
|   └── prepare_data.py
Then run this script to prepare the data of IAM
"""
import sys

sys.path.extend(['..'])
import numpy as np
import pickle as pkl
import cv2


def read_image(img_path, label_len, img_h=32, char_w=16):
    valid_img = True
    img = cv2.imread(img_path, 0)
    try:
        curr_h, curr_w = img.shape
        modified_w = int(curr_w * (img_h / curr_h))

        # Remove outliers
        if ((modified_w / label_len) < (char_w / 3)) | ((modified_w / label_len) > (3 * char_w)):
            valid_img = False
        else:
            # Resize image so height = img_h and width = char_w * label_len
            img_w = label_len * char_w
            img = cv2.resize(img, (img_w, img_h))

    except AttributeError:
        valid_img = False

    return img, valid_img


def read_data(config):
    """
    Saves dictionary of preprocessed images and labels for the required partition
    """
    img_h = config.img_h
    char_w = config.char_w
    partition = config.partition
    out_name = config.data_file
    data_folder_path = config.data_folder_path
    dataset = config.dataset

    if dataset == 'IAM':
        # Extract IDs for test, train and val sets
        with open(data_folder_path + '/original_partition/tr.lst', 'rb') as f:
            ids = f.read().decode('unicode_escape')
            train_ids = [i for i in ids.splitlines()]
        with open(data_folder_path + '/original_partition/va1.lst', 'rb') as f:
            ids = f.read().decode('unicode_escape')
            val_ids = [i for i in ids.splitlines()]
        with open(data_folder_path + '/original_partition/va2.lst', 'rb') as f:
            ids = f.read().decode('unicode_escape')
            val_ids += [i for i in ids.splitlines()]
        with open(data_folder_path + '/original_partition/te.lst', 'rb') as f:
            ids = f.read().decode('unicode_escape')
            test_ids = [i for i in ids.splitlines()]

        # Read labels and filter out the ones which just contain punctuation
        with open(data_folder_path + '/ascii/words.txt', 'rb') as f:
            char = f.read().decode('unicode_escape')
            words_raw = char.splitlines()[18:]

        punc_list = ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']
        # Get list of unique characters and create dictionary for mapping them to integer
        chars = np.unique(np.concatenate([[char for char in w_i.split()[-1] if w_i.split()[-1] not in punc_list]
                                          for w_i in words_raw]))
        char_map = {value: idx + 1 for (idx, value) in enumerate(chars)}
        char_map['<BLANK>'] = 0
        num_chars = len(char_map.keys())

        word_data = {}
        for word in words_raw:
            if word.split()[-1] not in punc_list:
                img_id = word.split()[0]
                label = word.split()[-1]

                if partition == 'tr':
                    partition_ids = train_ids
                elif partition == 'vl':
                    partition_ids = val_ids
                else:
                    partition_ids = test_ids

                if img_id[:img_id.rfind('-')] in partition_ids:
                    dir_data = img_id.split('-')
                    img_path = f'{data_folder_path}/words/{dir_data[0]}/{dir_data[0]}-{dir_data[1]}/{img_id}.png'
                    img, valid_img = read_image(img_path, len(label), img_h, char_w)
                    if valid_img:
                        word_data[img_id] = [[char_map[char] for char in label], img]

    else:
        if partition == 'tr':
            partition_name = 'training'
        elif partition == 'vl':
            partition_name = 'validation'
        else:
            partition_name = 'test'

        # create char_map using training labels
        with open(f'{data_folder_path}/ground_truth_training_icdar2011.txt', 'rb') as f:
            ids = f.read().decode('unicode_escape')
            partition_ids = [i.split()[0] for i in ids.splitlines() if len(i) > 1]
            words_raw = [i.split()[1] for i in ids.splitlines() if len(i) > 1]

        # Get list of unique characters and create dictionary for mapping them to integer
        chars = np.unique(np.concatenate([[char for char in w_i.split()[-1]] for w_i in words_raw]))
        char_map = {value: idx + 1 for (idx, value) in enumerate(chars)}
        char_map['<BLANK>'] = 0
        num_chars = len(char_map.keys())

        # Extract IDs for required set
        with open(f'{data_folder_path}/ground_truth_{partition_name}_icdar2011.txt', 'rb') as f:
            ids = f.read().decode('unicode_escape')
            partition_ids = [i.split()[0] for i in ids.splitlines() if len(i) > 1]
            words_raw = [i.split()[1] for i in ids.splitlines() if len(i) > 1]

        word_data = {}
        for img_path, label in zip(partition_ids, words_raw):
            img_path = f'{data_folder_path}/{partition_name}/{img_path}'
            img, valid_img = read_image(img_path, len(label), img_h, char_w)
            img_id = img_path[img_path.rfind('/')+1:-5]
            if valid_img:
                try:
                    word_data[img_id] = [[char_map[char] for char in label], img]
                except KeyError:
                    pass

    print(f'Number of images = {len(word_data)}')
    print(f'Number of unique characters = {num_chars}')

    # Save the data
    with open(f'{out_name[out_name.rfind("/")+1:]}', 'wb') as f:
        pkl.dump({'word_data': word_data,
                  'char_map': char_map,
                  'num_chars': num_chars}, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    from config import Config
    config = Config
    print('Processing Data:\n')
    read_data(config)
    print('\nData processing completed')
