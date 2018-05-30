# -*- coding: utf-8 -*-

# John Feltrup
# Spring Quarter 2018
# CSE 517 Natural Language Processing
#
# Test Test simpilest model, so we have something to turn in

import sys
import random
import pickle
import math

def main(argv):
    # Seed the random number generator
    random.seed(argv[0])

    # De-pickle the models
    unigramProbs = pickle.load(open("unigram_pickle.p", "rb"))

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
            #print nextUni

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
            logProb = math.log(unigramProbs[nextUni], 2)

            # Write the output to standardout
            sys.stdout.write(str(logProb))
            sys.stdout.write("\n")

            # then process the next character
            index += 1
            command = splitLine[index]
        elif command == 'g':
            # Randomly generate the next character given the distribution

            # Now, generate a character based on the model
            nextUni = None
            nextUni = unigramGenerator(unigramProbs)

            # Then get the log prob for that unicode character
            logProb = math.log(unigramProbs[nextUni], 2)

            # Write the character and log probability to standard out
            charToPrint = nextUni
            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write(charToPrint)
            sys.stdout.write(u"// Generated a character! Probability of generation " + str(logProb))
            sys.stdout.write(u"\n")

            # Get the next command
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


# Generates the next character based on a unigram model and the given history
def unigramGenerator(unigramProbs):
    # Spread the unigram choices between 0 and 1
    unigramLine = []
    firstGram = True
    index = 0
    probSum = 0
    for gram in unigramProbs:
        if firstGram:
            firstGram = False
            unigramLine.append((gram, probSum))
        else:
            unigramLine.append((gram, probSum))
        probSum += unigramProbs[gram]
        index += 1

    # Generate a random number between 0 and 1. Use this number to select a character
    randNum = random.random()
    nextUni = None
    for i in range(0, len(unigramLine) - 1):
        if randNum >= unigramLine[i][1] and randNum < unigramLine[i + 1][1]:
            nextUni = unigramLine[i][0]
    if nextUni == None:
        nextUni = unigramLine[len(unigramLine) - 1][0]
    return nextUni

if __name__ == '__main__':
    main(sys.argv[1:])