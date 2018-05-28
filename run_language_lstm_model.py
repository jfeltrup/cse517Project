# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:34:43 2018

@author: jamie
@author: John Feltrup 
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
    with open("DerivedNames.txt", encoding='utf-8') as f:
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
        return total_vocab


# Declaring vocabulary variables to be used as globals
vocabulary = buildVocabulary()
vocab_size = len(vocabulary)

INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
HIDDEN_DIM = 50 # 50 is the size of the largest model

# Path for saving/loading the model
MODEL_PATH = "saved_model_large.p"

# Defines the max length of the history (currently not being used)
HISTORY_LENGTH = 10

def main(argv):
    # Seed the random number generator
    random.seed(argv[0])

    # Make the model, then load only the parameters
    model = RNN_LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))

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
            
            # If it is the stop symbol, clear the history (Just the start character)
            if nextUni == chr(3):
                history = [chr(2)]
            else: # Otherwise append the character to the history
                history.append(nextUni)
            # TODO: If the size of history becomes a problem, we can put a length cap on it
            # if len(history) > HISTORY_LENGTH:
            #     del history[0]
            
            # Output to stdout so we know what is going on
            charToPrint = nextUni
            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write("// Added the character " + charToPrint + " to the history")
            sys.stdout.write("\n")

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
        
            # Add the character to the history, turn it into a tensor, and run the model
            history.append(nextUni)
            input = inputTensor(history)
            output = model(input)

            # Now, fetch the log probability for that character
            index = CharacterToIndex(nextUni)
            logProb = output[len(history) - 1][index]
            # Then, convert that log to log base 2
            logProb = logProb / math.log(2)
            
            # Write the output to standardout (it needs .data.numpy() for formatting)
            sys.stdout.write(str(logProb.data.numpy()))
            sys.stdout.write("\n")

            # Then make sure that the character is removed from the history
            history = history[:-1]
    
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
            # Using the random number, pick the character it generates
            # TODO: If this is too slow, then we can just pick the top 100-ish and go from there
            for i in range(0, len(output[len(history) - 1])):
                if cdf >= randNum:
                    break
                else:
                    selectedIndex += 1
                    cdf += math.pow(math.e, output[len(history) - 1][i])

            # Get the new character
            nextUni = vocabulary[selectedIndex]
            charToPrint = nextUni
            # Get the log prob, and calculate the base 2 log prob
            logProb = output[len(history) - 1][selectedIndex]
            logProb = logProb / math.log(2)

            # Print the character along with the log base 2 prob of generating it
            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write(charToPrint)
            sys.stdout.write(u"// Generated a character! Probability of generation " + str(logProb.data.numpy()))
            sys.stdout.write(u"\n")
            # Add the generated character to the history
            history.append(nextUni)
            # if len(history) > HISTORY_LENGTH:
            #     del history[0]
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
        output = self.h2o(lstm_out.view(len(input), -1))
        # output = self.dropout(output)
        output = self.softmax(output)
        return output


if __name__ == '__main__':
    main(sys.argv[1:])
