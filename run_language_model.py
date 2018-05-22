#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:34:43 2018

@author: jamie
"""
import torch
import sys
import random
import pickle
import math
import numpy as np

def main(argv):
    
    useNamed = True
    vocabulary = buildVocabulary(useNamed)
    vocab_size = len(vocabulary)
    
    #    print ("Load the model")
    model = torch.load("lstm_model.pth")

    history = []
    # Read in the input, make sure to treat it as unicode
    line = sys.stdin.readline()
    #line = unicode(line, "utf-8")
    splitLine = list(line)
    index = 0
    command = splitLine[index]
    # Process the command
    while command != 'x':
        if command == 'o':
            # get the next unicode character and add it to history
            index += 1
            nextUni = splitLine[index]
            # Special case, if the character is a newline, need to update the input line
            if nextUni == "\n":
                line = sys.stdin.readline()
                #line = unicode(line, "utf-8")
                splitLine = list(line)
                index = -1
            
            # If it is the stop symbol, clear the history
            if nextUni == chr(3):
                history = []
            else: # Otherwise append the character to the history
                history.append(nextUni)
            
            # Output to stdout so we know what is going on
            charToPrint = nextUni
            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write("// Added the character " + charToPrint + " to the history")
            sys.stdout.write("\n")
            
            # Then get the next command
            index += 1
        command = splitLine[index]
        elif command == 'q':
            # Give the base 2 log probability of the next character
            index += 1
            nextUni = splitLine[index]
            # Special case, if the character is a newline, need to update the input line
            if nextUni == "\n":
                line = sys.stdin.readline()
                splitLine = list(line)
                index = -1
        
            # Now, we calculate the probability for each model
            Prob = model(history)
            logProb=math.log(Prob, 2)
            
            index = CharacterToIndex(nextUni)
            
            # Write the output to standardout
            sys.stdout.write(str(logProb[index]))
            sys.stdout.write("\n")
    
            # then process the next character
            index += 1

        command = splitLine[index]

        elif command == 'g':
            # Randomly generate the next character given the distribution
            Prob = model(history)

            randNum = random.random()
            selectedIndex = 0
            for i in range(0, len(Prob)):
                cdf = 0
                cdf += Prob[i]
                if randNum >= cdf:
                    selectedIndex += 1
                else:
                    break

            nextUni = vocabulary[selectedIndex]
            charToPrint = nextUni

            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write(charToPrint)
            sys.stdout.write(u"// Generated a character! Probability of generation " + str(logProb))
            sys.stdout.write(u"\n")
            history.append(nextUni)
            index += 1
        command = splitLine[index]
        
        elif command == "x":
            # Command to exit program
            break
            
        else:
            # It is not a valid command
            sys.stdout.write("Command unrecognized. Exiting Program\n")
            print (command)
            break


# Returns a list of all the unicode characters that are a part of our valid vocabulary
# If "useNamed" is true, uses DerivedNames.txt
# If "useNamed" is false, uses Blocks.txt
# Both text files skip lines that start with "#" or "\n"
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

def CharacterToIndex(character):
    return vocabulary.find(character)


if __name__ == '__main__':
    main(sys.argv[1:])
