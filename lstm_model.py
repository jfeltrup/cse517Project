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
#FILE_PATH

# Prepare the data

def buildVocabulary(useNamed):
    if useNamed:
        total_vocab = []
        # First, we have to manually add control characters 0-31 (they are not a part of the named characters file)
        for i in range(0, 32):
            total_vocab.append(chr(i))
        # Second, we add the control characters from 007F - 009F
        for i in range(127, 160):
            total_vocab.append(chr(i))
        # Next read the named characters from the file
        with open("DerivedNames.txt") as f:
            line = f.readline()
            while line:
                if line.startswith("#") or line.startswith("\n"):
                    line = f.readline()
                else:
                    split1 = line.split(";")
                    split1 = split1[0].split(" ")
                    # There are two cases, either a single character or a range
                    # This first case is the range of values
                    if ".." in split1[0]:
                        split2 = split1[0].split("..")
                        lower = int(split2[0], 16)
                        upper = int(split2[1], 16)
                        for i in range(lower, upper + 1):
                            total_vocab.append(chr(i))
                    else: # The second case is a single value
                        number = int(split1[0], 16)
                        total_vocab.append(chr(number))
                    line = f.readline()
            # For testing purposes, manually adding 7F
            print(len(total_vocab))
            return total_vocab
    else:
        total_vocab = []
        with open("Blocks.txt") as f:
            line = f.readline()
            while line:
                if line.startswith("#") or line.startswith("\n"):
                    line = f.readline()
                else:
                    split1 = line.split(";")
                    split2 = split1[0].split("..")
                    lower = int(split2[0], 16)
                    upper = int(split2[1], 16)
                    for i in range(lower, upper+1):
                        total_vocab.append(chr(i))
                    line = f.readline()
            return total_vocab

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

def main():
    print ("Start of program")
    
    useNamed = True
    vocabulary = buildVocabulary(useNamed)
    vocab_size = len(vocabulary)
    
    model = RNN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)


    ## preprocess data
    print ("Read and preprocess the data")
    
    text_lines = readLines(FILE_PATH)
    tensors = []
    for line in text_lines:
        line_tensor = lineToTensor(line)
        tensors.append(line_tensor)

    ## Train the model
    print ("Train the model")

    for epoch in range(n_epoch):
        for tensor in tensors:
            # Clear the accumulates gradients out before each instance
            model.zero_grad()
            
            # Clear out the hidden state of the LSTM,
            model.hidden = model.init_hidden()
            
            # Run our forward pass.
            probabilities = model(tensor)
            
            # Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            ###########this step is not finished
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    ## Save the model
    print ("Save the model")
    torch.save(model, "lstm_model.pth")

if __name__ == '__main__':
    main()

