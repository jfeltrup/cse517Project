#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:34:43 2018

@author: jamie
"""
import torch
import torch.nn as nn
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string

# model

#INPUT_DIM = n_character
#OUTPUT_DIM = n_character
#HIDDEN_DIM = TBD
#n_epoch = 50


# Prepare the data
def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))


vocabulary = vocabulary # the vocabulary we built
vocab_size = len(vocabulary)

## read the data

### Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]

### Turn characters into tensors
def CharacterToIndex(character):
    return vocabulary.find(character)

### Turn a line into an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, vocab_size)
    for li, character in enumerate(line):
        tensor[li][0][CharacterToIndex(character)] = 1
    return tensor

# Define the model

class RNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        # Should we do this?
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, text):
        #text: character-wise input
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(text), 1, -1), self.hidden)
        output = self.h2o(lstm_out.view(len(text), -1))
        #output = self.dropout(output)
        output = self.softmax(output)
        return output


# Train the model

model = RNN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

## preprocess data


## Train the model

for epoch in range(n_epoch):
    for sentence in sentencess:
        # Clear the accumulates gradients out before each instance
        model.zero_grad()
        
        # Clear out the hidden state of the LSTM,
        model.hidden = model.init_hidden()

        # Run our forward pass.
        probabilities = model(sentences)
        
        # Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        ###########this step is not finished
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

