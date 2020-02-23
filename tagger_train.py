# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
import time
from random import uniform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from numpy.random import randint


# class CNN(nn.Module):


class LSTMTagger(nn.Module):
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    K = 3

    WORD_EMBEDDING_DIM = 1024
    CHAR_EMBEDDING_DIM = 128
    WORD_HIDDEN_DIM = 1024
    CHAR_HIDDEN_DIM = 1024
    EPOCHS = 70

    @staticmethod
    def prepare_sequence(seq, to_ix):
        idxs = []
        for w in seq:
            if w in to_ix:
                idx = to_ix[w]
            else:
                idx = randint(len(to_ix))
            idxs.append(idx)
        idxs = torch.tensor(idxs, dtype=torch.long)
        if USE_CUDA and torch.cuda.is_available():
            idxs = idxs.cuda()
        return idxs

    def pad_word(self, word):
        word=list(word)
        left_right_pad_count = int((self.K - 1) / 2)
        for i in range(left_right_pad_count):
            word.insert(0, UNUSED_CHAR)
            word.append(UNUSED_CHAR)
        return word

    def __init__(self, word_to_idx, tag_to_idx, char_to_idx, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super(LSTMTagger, self).__init__()
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.char_to_idx = char_to_idx
        self.hidden_dim = hidden_dim

        vocab_size = len(word_to_idx)
        tagset_size = len(tag_to_idx)
        alphabet_size = len(char_to_idx)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        in_channel = 1
        l = 4
        dw = 1
        self.lstm = nn.LSTM(embedding_dim+l, hidden_dim,bidirectional=True)
        self.cnn = nn.Conv1d(in_channel, l, self.K*embedding_dim, dw)
        # self.cnn = nn.MaxPool1d(kernel_size=2, stride=2)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        # Character level representation for words
        char_reprs = []
        for word in sentence:
            word = self.pad_word(word)
            word = self.prepare_sequence(word, self.char_to_idx)
            word_char_embeds = self.char_embeddings(word)
            left_right_surrounding = int((self.K - 1) / 2)
            x_hats=[]
            for i in range(left_right_surrounding, len(word_char_embeds) - left_right_surrounding):
                x_hat=word_char_embeds[i-left_right_surrounding:i+left_right_surrounding+1]
                x_hat=torch.flatten(x_hat)
                x_hat=torch.reshape(x_hat, (1,1,-1))
                x_hats.append(x_hat)
            x_hats=torch.cat(x_hats,0)

            phi=self.cnn(x_hats)
            phi=torch.squeeze(phi,-1)
            char_repr = torch.max(phi,0)[0]
            char_repr = torch.unsqueeze(char_repr, 0)
            char_reprs.append(char_repr)
        char_reprs = torch.cat(char_reprs, 0)

        # Embedding vector for words
        sentence = self.prepare_sequence(sentence, self.word_to_idx)
        word_embeds = self.word_embeddings(sentence)

        combined = torch.cat((word_embeds, char_reprs), 1)

        lstm_out, _ = self.lstm(combined.view(len(sentence), 1, -1))
        s1=lstm_out.shape
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# Use CUDA for training on GPU
USE_CUDA = False
NUM_EPOCHS = 1
UNUSED_CHAR = '∞'


# inp=5  # dimensionality of one sequence element
# outp=1 # number of derived features for one sequence element
# kw=3   # kernel only operates on one sequence element per step
# dw=1   # we step once and go on to the next sequence element
#
# mlp=nn.TemporalConvolution(inp,outp,kw,dw)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    idxs = torch.tensor(idxs, dtype=torch.long)
    if USE_CUDA and torch.cuda.is_available():
        idxs = idxs.cuda()
    return idxs

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    with open(train_file) as f:
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

    model = LSTMTagger(word_to_idx, tag_to_idx, char_to_idx)
    if USE_CUDA and torch.cuda.is_available():
        model = model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    total_step = len(training_data)
    loss_list = []
    acc_list = []
    ### Reduce number of epochs, if training data is big
    start_time=time.time()
    for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        for i, (sentence, tags) in enumerate(training_data):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 3. Run our forward pass.
            tag_scores = model(sentence)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            targets = prepare_sequence(tags, tag_to_idx)
            loss = loss_function(tag_scores, targets)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            total = targets.size(0)
            _, predicted = torch.max(tag_scores.data, 1)
            correct = (predicted == targets).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    print(f'Duration: {time.time()-start_time}')
    torch.save((word_to_idx, tag_to_idx, char_to_idx, model.state_dict()), model_file)
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
