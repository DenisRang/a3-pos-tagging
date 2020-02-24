# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>
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
    WORD_EMBEDDING_DIM = 10
    CHAR_EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    K = 3  # CNN sliding window size
    l = 4  # Number of different convolutional filters
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

    def __init__(self, word_to_idx, tag_to_idx, char_to_idx):
        super(LSTMTagger, self).__init__()
        self.char_to_idx = char_to_idx
        self.tag_to_idx = tag_to_idx
        self.word_to_idx = word_to_idx
        hidden_dim = self.HIDDEN_DIM

        vocab_size = len(word_to_idx)
        tagset_size = len(tag_to_idx)
        alphabet_size = len(char_to_idx)
        self.word_embeddings = nn.Embedding(vocab_size, self.WORD_EMBEDDING_DIM)
        self.char_embeddings = nn.Embedding(alphabet_size, self.CHAR_EMBEDDING_DIM)

        self.lstm = nn.LSTM(self.WORD_EMBEDDING_DIM + self.l, hidden_dim, bidirectional=True)
        self.cnn = nn.Conv1d(1, self.l, self.K * self.CHAR_EMBEDDING_DIM, 1)
        # self.drop_out = nn.Dropout(0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences, sentences_seq):
        char_embeds = []
        lengths = []
        maxx = 0
        tag_score_list = []
        
        for sentence in sentences:
            if maxx<len(sentence):
                maxx = len(sentence)
        
        for sentence in sentences:
            # Character level representation for words
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
            if maxx != len(sentence):
                temp = torch.cat((torch.zeros(maxx-len(sentence),134), torch.ones(maxx-len(sentence),1)),1)
                tag_scores = torch.cat((temp, tag_scores),0)
            tag_score_list.append(tag_scores)
            
        return torch.stack(tag_score_list)


class CorpusDataset(Dataset):
    def __init__(self, training_data, word_to_idx, tag_to_idx):
        self.training_data = training_data
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx

    def __getitem__(self, index):
        sent = self.training_data[index][0]
        sent_in = self.prepare_sequence(sent, self.word_to_idx)
        tag_in = self.prepare_sequence(self.training_data[index][1], self.tag_to_idx)
        return (sent, sent_in, tag_in)

    def __len__(self):
        return len(self.training_data)

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


def pad_collate(batch):
    (sents, sents_in, tags_in) = zip(*batch)

    #sents_in_pad = pad_sequence(sents_in, batch_first=True, padding_value=padding_word_idx)
    tags_in_pad = pad_sequence(tags_in, batch_first=True, padding_value=padding_tag_idx)

    return sents, sents_in, tags_in_pad


USE_CUDA = False  # Use CUDA for training on GPU
UNUSED_ITEM = '∞'
NUM_EPOCHS = 1
BATCH_SIZE = 4
padding_word_idx = 0
padding_tag_idx = 0


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
    padding_word_idx = len(word_to_idx) - 1
    padding_tag_idx = len(tag_to_idx) - 1

    model = LSTMTagger(word_to_idx, tag_to_idx, char_to_idx)
    if USE_CUDA and torch.cuda.is_available():
        model = model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_data = CorpusDataset(training_data, word_to_idx, tag_to_idx)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data

        for item in tqdm(train_loader):
            # Clear gradients out before each instance
            model.zero_grad()
            sentence = item[0]
#             for i in range(len(item[0])):
#                 sent = item[0][i] + [UNUSED_ITEM for i in range(len(item[1][0]) - len(item[0][i]))]
#                 sentence.append(sent)

            # Forward pass.
            tag_scores = model(sentence, item[1])

            temp_loss = torch.Tensor([0])
            for i in range(len(tag_scores)):
                temp_loss += loss_function(tag_scores[i], item[2][i])

            loss = temp_loss / BATCH_SIZE
            print(f'  Loss: {float(loss):.4f}')
            loss.backward()
            optimizer.step()

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
