#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:34:43 2018

@author: jamie
"""
import torch
import torch.nn as nn
import sys
import random
import pickle
import math
import numpy as np

# Returns a list of all the unicode characters that are a part of our valid vocabulary
# Both text files skip lines that start with "#" or "\n"
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
                else: # The second case is a single value
                    number = int(split1[0], 16)
                    total_vocab.append(chr(number))
                line = f.readline()
        # For testing purposes, manually adding 7F
        print(len(total_vocab))
        return total_vocab


# Declaring vocabulary variables to be used as globals
vocabulary = buildVocabulary()
vocab_size = 136755

INPUT_DIM = 136755
OUTPUT_DIM = 136755
HIDDEN_DIM = 10  # TBD

def main(argv):

    # TEMP
    print("Starting program")

    vocab_size = len(vocabulary)

    # TEMP
    print("Loading Model")
    
    #    print ("Load the model")
    model = torch.load("lstm_model_save.p")

    # Loads only the model parameters
    #model = RNN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    # TEMP
    print("Loaded Model")

    history = [chr(2)]
    # Read in the input, make sure to treat it as unicode
    line = sys.stdin.readline()
    #line = unicode(line, "utf-8")
    splitLine = list(line)
    commandIndex = 0
    command = splitLine[commandIndex]
    # Process the command
    while command != 'x':
        if command == 'o':
            # get the next unicode character and add it to history
            commandIndex += 1
            nextUni = splitLine[commandIndex]
            # Special case, if the character is a newline, need to update the input line
            if nextUni == "\n":
                line = sys.stdin.readline()
                #line = unicode(line, "utf-8")
                splitLine = list(line)
                commandIndex = -1
            
            # If it is the stop symbol, clear the history
            if nextUni == chr(3):
                history = [chr(2)]
            else: # Otherwise append the character to the history
                history.append(nextUni)
            
            # Output to stdout so we know what is going on
            charToPrint = nextUni
            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write("// Added the character " + charToPrint + " to the history")
            sys.stdout.write("\n")

            print("History is now: " + str(history))

            # TODO: Maybe put a cap on the size of the history? cut if off at a certain point
            
            # Then get the next command
            commandIndex += 1
            command = splitLine[commandIndex]
        elif command == 'q':
            # Give the base 2 log probability of the next character
            commandIndex += 1
            nextUni = splitLine[commandIndex]
            # Special case, if the character is a newline, need to update the input line
            if nextUni == "\n":
                line = sys.stdin.readline()
                splitLine = list(line)
                commandIndex = -1
        
            # Now, we calculate the probability for each model
            # Prob = model(history)
            # logProb=math.log(Prob, 2)
            #
            # index = CharacterToIndex(nextUni)

            # This is a test. Turn the history into an input tensor then try it
            # Okay, good to know that this works

            # Add the character to the history, turn it into a tensor, and run the model
            history.append(nextUni)
            input = inputTensor(history)
            output = model(input)

            # Now, fetch the log probability for that character
            index = CharacterToIndex(nextUni)
            logProb = output[len(history) - 1][index]
            print("This is the log probability of the character: " + str(logProb))
            # Then, convert that log to log base 2
            logProb = logProb / math.log(2)
            print("This is the log base 2 probability: " + str(logProb))

            # This is a little excessive for the formatting, but if it works I won't complain
            print("This is the value formatted: " + str(logProb.data.numpy()))
            # Then print it out
            
            # Write the output to standardout
            sys.stdout.write(str(logProb.data.numpy()))
            sys.stdout.write("\n")

            # Then make sure that the character is removed from the history
            history = history[:-1]

            print("updated history: " + str(history))
    
            # then process the next character
            commandIndex += 1

            command = splitLine[commandIndex]

        elif command == 'g':
            # Randomly generate the next character given the distribution
            input = inputTensor(history)
            output = model(input)

            randNum = random.random()
            selectedIndex = 0
            cdf = 0
            # TODO: Does this work? I really don't know
            print("this is the original random number: " + str(randNum))
            #randNum = math.log(randNum)
            #print("this is the log of the random number: " + str(randNum))
            #print("this is how we undo the log prob: " + str(math.pow(math.e, randNum)))
            for i in range(0, len(output[len(history) - 1])):
                if cdf >= randNum:
                    break
                else:
                    selectedIndex += 1
                    cdf += math.pow(math.e, output[len(history) - 1][i])

            print("this is the selected index: " + str(selectedIndex))
            nextUni = vocabulary[selectedIndex]
            charToPrint = nextUni

            logProb = output[len(history) - 1][selectedIndex]
            print("This is the log probability of the character: " + str(logProb))
            logProb = logProb / math.log(2)
            print("This is the log base 2 probability: " + str(logProb))

            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write(charToPrint)
            sys.stdout.write(u"// Generated a character! Probability of generation " + str(logProb.data.numpy()))
            sys.stdout.write(u"\n")
            history.append(nextUni)
            commandIndex += 1
            command = splitLine[commandIndex]
        elif command == "x":
            # Command to exit program
            break
        else:
            # It is not a valid command
            sys.stdout.write("Command unrecognized. Exiting Program\n")
            print (command)
            break

### Turn characters into tensors
def CharacterToIndex(character):
    return vocabulary.index(character)

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, vocab_size)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][vocabulary.index(letter)] = 1
    return tensor


# Define the model
class RNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        # self.o2o = nn.Linear(hidden_dim + output_dim, output_dim)
        # Should we do this?
        # self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, input):
        # Character input one hot vector
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        # TODO: Double check these lines. I replaced the code using "embed" so I am not 100%
        # sure that is is what we want to do
        output = self.h2o(lstm_out.view(len(input), -1))
        # output = self.dropout(output)
        output = self.softmax(output)
        return output


if __name__ == '__main__':
    main(sys.argv[1:])
