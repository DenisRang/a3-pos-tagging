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
from tqdm import tqdm
from numpy.random import randint
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTMTagger(nn.Module):
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    K = 3


    WORD_EMBEDDING_DIM = 10
    CHAR_EMBEDDING_DIM = 6
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
        word = list(word)
        left_right_pad_count = int((self.K - 1) / 2)
        for i in range(left_right_pad_count):
            word.insert(0, UNUSED_ITEM)
            word.append(UNUSED_ITEM)
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
        self.word_embeddings = nn.Embedding(vocab_size, self.WORD_EMBEDDING_DIM)
        self.char_embeddings = nn.Embedding(alphabet_size, self.CHAR_EMBEDDING_DIM)

        in_channel = 1
        l = 4
        dw = 1
        self.lstm = nn.LSTM(self.WORD_EMBEDDING_DIM+l, hidden_dim, bidirectional=True)
        self.cnn = nn.Conv1d(in_channel, l, self.K * self.CHAR_EMBEDDING_DIM, dw)
        # self.drop_out = nn.Dropout(0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences, sentences_seq):
        char_embeds = []
        lengths = []
        for sentence in sentences:
            # Character level representation for words
            lengths.append(len(sentence))
            char_reprs = []
            for word in sentence:
                word = self.pad_word(word)
                word = self.prepare_sequence(word, self.char_to_idx)
                word_char_embeds = self.char_embeddings(word)
                left_right_surrounding = int((self.K - 1) / 2)
                x_hats = []
                for i in range(left_right_surrounding, len(word_char_embeds) - left_right_surrounding):
                    x_hat = word_char_embeds[i - left_right_surrounding:i + left_right_surrounding + 1]
                    x_hat = torch.flatten(x_hat)
                    x_hat = torch.reshape(x_hat, (1, 1, -1))
                    x_hats.append(x_hat)
                x_hats = torch.cat(x_hats, 0)

                phi = self.cnn(x_hats)
                phi = torch.squeeze(phi, -1)
                char_repr = torch.max(phi, 0)[0]
                char_repr = torch.unsqueeze(char_repr, 0)
                char_reprs.append(char_repr)
            char_reprs = torch.cat(char_reprs, 0)
            char_embeds.append(char_reprs)

        char_embeds = torch.stack(char_embeds)

        word_embeds = self.word_embeddings(sentences_seq)
        combined = torch.cat((word_embeds, char_embeds), 2)
        lstm_out, _ = self.lstm(combined.view(len(combined[0]), len(combined), -1))
        # lstm_out = self.drop_out(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentences_seq), len(sentences_seq[0]), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# class CNN(nn.Module):
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    idxs = torch.tensor(idxs, dtype=torch.long)
    if USE_CUDA and torch.cuda.is_available():
        idxs = idxs.cuda()
    return idxs


class CustomDataset(Dataset):
    def __init__(self, training_data, word_to_idx, tag_to_idx):
        self.training_data = training_data
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

    def __getitem__(self, index):
        sent = self.training_data[index][0]
        sent_in = prepare_sequence(sent, self.word_to_idx)
        tag_in = prepare_sequence(self.training_data[index][1], self.tag_to_idx)
        return (sent, sent_in, tag_in)

    def __len__(self):
        return len(self.training_data)


def pad_collate(batch):
    (sents, sents_in, tags_in) = zip(*batch)

    sents_in_pad = pad_sequence(sents_in, batch_first=True, padding_value=43057)
    tags_in_pad = pad_sequence(tags_in, batch_first=True, padding_value=134)

    return sents, sents_in_pad, tags_in_pad


# Use CUDA for training on GPU
USE_CUDA = False
NUM_EPOCHS = 1
UNUSED_ITEM = '∞'
BATCH_SIZE = 128


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file

    with open(train_file) as f:
        content = f.read().splitlines()

    content.sort(key=lambda x: len(x.split()), reverse=False)

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
    word_to_idx[UNUSED_ITEM] = len(word_to_idx)
    tag_to_idx[UNUSED_ITEM] = len(tag_to_idx)
    char_to_idx[UNUSED_ITEM] = len(char_to_idx)


    model = LSTMTagger(word_to_idx, tag_to_idx, char_to_idx)
    if USE_CUDA and torch.cuda.is_available():
        model = model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=10)

    total_step = len(training_data)
    loss_list = []
    acc_list = []
    ### Reduce number of epochs, if training data is big
    start_time = time.time()

    train_data = CustomDataset(training_data, word_to_idx, tag_to_idx)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate,num_workers=BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data

        for item in tqdm(train_loader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            sentence = []
            for i in range(len(item[0])):
                sent = item[0][i] + [UNUSED_ITEM for i in range(len(item[1][0]) - len(item[0][i]))]
                sentence.append(sent)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence, item[1])
            #
            # # Step 4. Compute the loss, gradients, and update the parameters by
            # #  calling optimizer.step()
            # targets = prepare_sequence(tags, tag_to_idx)
            temp_loss = torch.Tensor([0])
            for i in range(len(tag_scores)):
                temp_loss += loss_function(tag_scores[i], item[2][i])

            loss = temp_loss / BATCH_SIZE
            # print('Loss: ' + str(loss.item()))
            #         loss_list.append(loss.item())
            print(loss)
            loss.backward()
            optimizer.step()
            #
        #         correct = (predicted == targets).sum().item()
        #         acc_list.append(correct / total)

        #         if (i + 1) % 100 == 0:
        #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
        #                   .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item(),
        #                           (correct / total) * 100))
        print("Epoch: " + str(epoch))
        print("Loss: " + str(loss))
    print(f'Duration: {time.time() - start_time}')
    torch.save((word_to_idx, tag_to_idx, char_to_idx, model.state_dict()), model_file)
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
