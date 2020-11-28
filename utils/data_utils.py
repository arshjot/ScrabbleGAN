import torch
from torch.nn.utils.rnn import pad_sequence


class WordMap:
    """Class for encoding and decoding word (str) to word(int) (and vice versa) using char_map"""

    def __init__(self, char_map):
        self.char_map = char_map
        self.rev_char_map = {val: key for key, val in char_map.items()}

    def encode(self, word_list):
        """Returns a padded encoded sequence of ints and an array of word's actual lengths"""

        enc_words, word_lens = [], []
        for word in word_list:
            enc_words.append(torch.LongTensor([self.char_map[char] for char in word]))
            word_lens.append(len(word))

        enc_pad_words = pad_sequence(enc_words, batch_first=True, padding_value=0)

        return enc_pad_words, torch.LongTensor(word_lens)

    def decode(self, enc_word_list):
        """Returns a list of words (str)"""

        dec_words = []
        for word in enc_word_list:
            dec_words.append(''.join([self.rev_char_map[char_enc] for char_enc in word if char_enc != 0]))

        return dec_words

    def recognizer_decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing repeating characters"""

        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol or repeated characters
                if (char_enc != 0) and (not ((idx > 0) and (char_enc == word[idx - 1]))):
                    word_chars += self.rev_char_map[char_enc]

            dec_words.append(word_chars)

        return dec_words
