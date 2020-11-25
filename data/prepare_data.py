"""
This file is for preparing IAM word level dataset
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
|   └── prepare_data.py
Then run this script to prepare the data of IAM
"""
import sys

sys.path.extend(['..'])
import numpy as np
import pickle as pkl
import cv2


def read_image(data_folder_path, img_id, label_len, img_h=32, char_w=16):
    valid_img = True
    dir_data = img_id.split('-')
    img = cv2.imread(f'{data_folder_path}/words/{dir_data[0]}/{dir_data[0]}-{dir_data[1]}/{img_id}.png', 0)
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
                img, valid_img = read_image(data_folder_path, img_id, len(label), img_h, char_w)
                if valid_img:
                    word_data[img_id] = [[char_map[char] for char in label], img]

    print(f'Number of images = {len(word_data)}')

    # Save the data
    with open(f'{out_name}', 'wb') as f:
        pkl.dump({'word_data': word_data,
                  'char_map': char_map,
                  'num_chars': num_chars}, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    from config import Config
    config = Config
    print('Processing Data:\n')
    read_data(config)
    print('\nData processing completed')
