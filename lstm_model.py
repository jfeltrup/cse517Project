# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:34:43 2018

@author: jamie
@author: John Feltrup
"""
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from io import open

USE_GPU = True
# Set up the GPU part of the code
device = None
if USE_GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Prepare the vocabulary using unicode files
def buildVocabulary():
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
                else:  # The second case is a single value
                    number = int(split1[0], 16)
                    total_vocab.append(chr(number))
                line = f.readline()
        # For testing purposes, manually adding 7F
        print("Vocab size: " + str(len(total_vocab)))
        return total_vocab


# File path for training data
FILE_PATH = "small_dataset_5.txt"

# Path for saving/loading the model
MODEL_PATH = "lstm_model_save.p"

# Declaring vocabulary variables to be used as globals
vocabulary = buildVocabulary()
vocab_size = len(vocabulary)

# Model Parameters
INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
HIDDEN_DIM = 10  # TBD
n_epoch = 5  # 50

# Use this variable to determine whether we want to create a new lstm_model, or load it from a file
LOAD_MODEL = False


# Train the model
def main():
    print("Start of program")

    print("Building Model")
    model = None
    if LOAD_MODEL:
        model = RNN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device=device)
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model = RNN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device=device)
    print("Finished Building Model")
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    ## preprocess data
    print("Read and preprocess the data")

    text_lines = readLines(FILE_PATH)
    training_data = []

    for line in text_lines:
        training_data.append((inputTensor(line), targetTensor(line)))

    ## Train the model
    print("Train the model")

    for epoch in range(n_epoch):
        #print the current epoch
        print("this is epoch:" + str(epoch))
        count = 0
        for char, next_char in training_data:
            # Print the current line of the file we are reading
            count += 1
            print("line count: " + str(count))
            # Clear the accumulates gradients out before each instance
            model.zero_grad()

            # Clear out the hidden state of the LSTM,
            model.hidden = model.init_hidden()

            # Run our forward pass.
            next_char_prob = model(char)

            # Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(next_char_prob, next_char)
            loss.backward()
            optimizer.step()

    ## Save the model
    print("Save the model parameters")
    # Saves only model parameters
    torch.save(model.state_dict(), MODEL_PATH)


### Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]


### Turn characters into tensors
def CharacterToIndex(character):
    return vocabulary.index(character)


### Turn a line into an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, vocab_size).to(device=device)
    for li, character in enumerate(line):
        tensor[li][0][CharacterToIndex(character)] = 1
    return tensor


## prepare the data

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, vocab_size).to(device=device)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][vocabulary.index(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [vocabulary.index(line[li]) for li in range(1, len(line))]
    letter_indexes.append(3)  # EOS
    return torch.LongTensor(letter_indexes).to(device=device)


# Define the model
class RNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        # Should we do this?
        # self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(device=device),
                torch.zeros(1, 1, self.hidden_dim).to(device=device))

    def forward(self, input):
        # Character input one hot vector
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        output = self.h2o(lstm_out.view(len(input), -1))
        # output = self.dropout(output)
        output = self.softmax(output)
        return output


if __name__ == '__main__':
    main()

