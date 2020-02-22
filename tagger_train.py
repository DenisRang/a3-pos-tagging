# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from numpy.random import randint


USE_CUDA = False
NUM_EPOCHS = 1
UNUSED_CHAR = 'unused_char'
UNUSED_WORD = 'unused_word' 
UNUSED_TAG = 'unused_tag'
PADDING_SIZE = 80
BATCH_SIZE = 64
K = 3


# inp=5  # dimensionality of one sequence element
# outp=1 # number of derived features for one sequence element
# kw=3   # kernel only operates on one sequence element per step
# dw=1   # we step once and go on to the next sequence element
#
# mlp=nn.TemporalConvolution(inp,outp,kw,dw)

def prepare_seq(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix:
            idx = to_ix[w]
        else:
            idx = randint(len(to_ix))
        idxs.append(idx)
    idxs = torch.tensor(idxs, dtype=torch.long)

    return idxs
    
    
def prepare_char_seq(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix:
            idx = to_ix[w]
        else:
            idx = randint(len(to_ix))
        idxs.append(idx)
    idxs = torch.tensor(idxs, dtype=torch.long)
    return idxs



def pad_word(word):
    word=list(word)
    left_right_pad_count = int((K - 1) / 2)
    for i in range(left_right_pad_count):
        word.insert(0, UNUSED_CHAR)
        word.append(UNUSED_CHAR)
    return word





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
        batch_idxs = []
        for item in seq:
            idxs = []
            for w in item:
                if w in to_ix:
                    idx = to_ix[w]
                else:
                    idx = randint(len(to_ix))
                idxs.append(idx)
            idxs = torch.tensor(idxs, dtype=torch.long)
            batch_idxs.append(idxs)
        
        return torch.stack(batch_idxs)
    
    @staticmethod
    def prepare_char_seq(seq, to_ix):
        idxs = []
        for w in seq:
            if w in to_ix:
                idx = to_ix[w]
            else:
                idx = randint(len(to_ix))
            idxs.append(idx)
        idxs = torch.tensor(idxs, dtype=torch.long)
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
        self.lstm = nn.LSTM(embedding_dim+l, hidden_dim)
        self.cnn = nn.Conv1d(in_channel, l, self.K*embedding_dim, dw)
        # self.cnn = nn.MaxPool1d(kernel_size=2, stride=2)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_batch, character_batch):
        # Character level representation for words
        #batch of char representation
        batch_char_reprs = []
        for sentence in character_batch:
            char_reprs = []
            for word in sentence:
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
            
            batch_char_reprs.append(char_reprs)
            
        batch_char_reprs = torch.stack(batch_char_reprs)
        
        # Embedding vector for words
        sentence_batch = self.prepare_sequence(sentence_batch, self.word_to_idx)
        word_embeds = self.word_embeddings(sentence_batch)
        
        combined = torch.cat((word_embeds, batch_char_reprs), 2)
        
        lstm_out, _ = self.lstm(combined.view(PADDING_SIZE, BATCH_SIZE, -1))
        tag_space = self.hidden2tag(lstm_out.view(BATCH_SIZE, PADDING_SIZE, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        
                
#         batch_char_reprs = []
#         for sentence in sentence_batch:
#             char_reprs = []
#             for word in sentence:
#                 word = self.pad_word(word)#can be done outsitde
#                 word = self.prepare_char_seq(word, self.char_to_idx)#can be done outsitde
#                 word_char_embeds = self.char_embeddings(word)
                
                
#                 left_right_surrounding = int((self.K - 1) / 2)
#                 x_hats=[]
#                 for i in range(left_right_surrounding, len(word_char_embeds) - left_right_surrounding):
#                     x_hat=word_char_embeds[i-left_right_surrounding:i+left_right_surrounding+1]
#                     x_hat=torch.flatten(x_hat)
#                     x_hat=torch.reshape(x_hat, (1,1,-1))
#                     x_hats.append(x_hat)
#                 x_hats=torch.cat(x_hats,0)

#                 phi=self.cnn(x_hats)
#                 phi=torch.squeeze(phi,-1)
#                 char_repr = torch.max(phi,0)[0]
#                 char_repr = torch.unsqueeze(char_repr, 0)
#                 char_reprs.append(char_repr)
            
        return tag_scores



def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    train_file = 'corpus.train'

    prep = time.time()
    print("Preprocessing...")
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
    word_to_idx[UNUSED_WORD] = len(word_to_idx)
    tag_to_idx[UNUSED_TAG] = len(tag_to_idx)


    #     print(word_to_idx)
    #     print(tag_to_idx)
    #     print(char_to_idx)


    model = LSTMTagger(word_to_idx, tag_to_idx, char_to_idx)
    if USE_CUDA and torch.cuda.is_available():
        model = model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    total_step = len(training_data)
    loss_list = []
    acc_list = []

    #BATCH
    sentence_batches = []
    tag_batches = []
    character_batches = []

    sen = []
    tag = []
    word_c = []
    sent_c = []
    counter = 0

    for i in range(len(training_data)):



        if BATCH_SIZE == counter:
            sen = torch.stack(sen)
            tag = torch.stack(tag)

            sentence_batches.append(sen)
            tag_batches.append(tag)
            character_batches.append(sent_c)


            counter = 0

            sen = []
            tag = []
            sent_c = []

        sen_temp = training_data[i][0] + ([UNUSED_WORD for i in range(PADDING_SIZE - len(training_data[i][0]))])
        word_c = []
        for word in sen_temp[:PADDING_SIZE]:
            word = pad_word(word)
            word = prepare_char_seq(word, char_to_idx)
            word_c.append(word)



        sen_temp = prepare_seq(sen_temp, word_to_idx)
        tag_temp = training_data[i][1] + ([UNUSED_TAG for i in range(PADDING_SIZE - len(training_data[i][1]))])
        tag_temp = prepare_seq(tag_temp, tag_to_idx)

        sent_c.append(word_c)
        sen.append(sen_temp[:PADDING_SIZE])
        tag.append(tag_temp[:PADDING_SIZE])


        counter = counter + 1

    train_time = time.time()
    print(train_time - prep)

    print("Training...")
    ### Reduce number of epochs, if training data is big
    for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        
            for batch in tqdm(range(len(sentence_batches))):
                model.zero_grad()
            
            
                tag_scores = model(sentence_batches[batch], character_batches[batch])
            
            
                temp_loss = torch.Tensor([0])
            
                for item in (range(len(tag_scores))):    
                    temp_loss += loss_function(tag_scores[item],tag_batches[batch][item])
            
                loss = temp_loss/BATCH_SIZE
                print(loss)
                loss.backward()
        
                optimizer.step()
            

    torch.save((word_to_idx, tag_to_idx, char_to_idx, model.state_dict()), model_file)
    print('Finished...')





if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
