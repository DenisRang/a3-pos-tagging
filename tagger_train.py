# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

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
    print(training_data[1])

    word_to_idx = {}
    tag_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
        for tag in tags:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)


    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[1][0], word_to_idx)
        tag_scores = model(inputs)
        print(tag_scores)

    ### Reduce number of epochs, if training data is big
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_idx)
            targets = prepare_sequence(tags, tag_to_idx)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        # print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))


    torch.save((word_to_idx, tag_to_idx, model.state_dict()), model_file)


    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
