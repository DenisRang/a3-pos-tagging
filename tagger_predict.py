# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import sys
import torch
import numpy as np
import pickle
from numpy.random import randint
from tagger_train import LSTMTagger

UNUSED_CHAR = 'unused'

def prepare_test_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix:
            idx = to_ix[w]
        else:
            idx = randint(len(to_ix))
        idxs.append(idx)
    return torch.tensor(idxs, dtype=torch.long)


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file


    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    with open("/Users/denisrangulov/Google Drive/Assignment-3/corpus.train") as f:
        content = f.read().splitlines()

    training_data = []
    for line in content:
        words = []
        tags = []
        for word_with_tag in line.split():
            word_with_tag_split = word_with_tag.split('/')
            word, tag = word_with_tag_split[0], word_with_tag_split[1]
            words.append(word)
            tags.append(tag)
        sample = (words, tags)
        training_data.append(sample)

    word_to_idx = {}
    tag_to_idx = {}
    char_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                for char in word:
                    if char not in char_to_idx:
                        char_to_idx[char] = len(char_to_idx)
        for tag in tags:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
    char_to_idx[UNUSED_CHAR] = len(char_to_idx)
    print(word_to_idx)
    print(tag_to_idx)
    print(char_to_idx)



    # word_to_idx, tag_to_idx, model_state_dict = torch.load(model_file)
    qqq, qq, model_state_dict = torch.load(model_file)
    model = LSTMTagger(word_to_idx, tag_to_idx, char_to_idx)
    model.load_state_dict(model_state_dict)

    with open(test_file) as f:
        content = f.read().splitlines()

    with open(out_file, 'w') as f:
        for sentence in content:
            sentence_in = prepare_test_sequence(sentence.split(), word_to_idx)
            # Forward pass
            tag_scores = model(sentence)
            for i in range(0, len(sentence_in)):
                word_idx = sentence_in[i]
                tag_idx = tag_scores[i].argmax()
                f.write(f'{list(word_to_idx.keys())[int(word_idx)]}/{list(tag_to_idx.keys())[int(tag_idx)]} ')
            f.write('\n')

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
