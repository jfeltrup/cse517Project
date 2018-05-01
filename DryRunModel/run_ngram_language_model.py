# -*- coding: utf-8 -*-

# John Feltrup
# Spring Quarter 2018
# CSE 517 Natural Language Processing

import sys
import random
import pickle
import math
# Code for testing the generation
#from format_check import is_valid_bmp

uniAlpha = 0.2
biAlpha = 0.1
triAlpha = 0.1
quadAlpha = 0.3
pentaAlpha = 0.3

def main(argv):
    # Seed the random number generator
    random.seed(argv[0])

    # De-pickle the models
    unigramProbs = pickle.load(open("unigram_pickle.p", "rb"))
    bigramProbs = pickle.load(open("bigram_pickle.p", "rb"))
    trigramProbs = pickle.load(open("trigram_pickle.p", "rb"))
    quadgramProbs = pickle.load(open("quadgram_pickle.p", "rb"))
    pentagramProbs = pickle.load(open("pentagram_pickle.p", "rb"))

    # Now, set up the history and reading from standard input
    history = [chr(2), chr(2), chr(2), chr(2)]

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

            # If it is the stop symbol, clear the history
            if nextUni == chr(3):
                history = [chr(2), chr(2), chr(2), chr(2)]
            else: # Otherwise append the character to the history
                del history[0]
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
            logProb = calculateLogLikelihood(nextUni, history, unigramProbs, bigramProbs, trigramProbs, quadgramProbs, pentagramProbs)

            # Write the output to standardout
            sys.stdout.write(str(logProb))
            sys.stdout.write("\n")

            # then process the next character
            index += 1
            command = splitLine[index]
        elif command == 'g':
            # Randomly generate the next character given the distribution

            # First, select a model to generate from based on the alpha values
            modelSelect = []
            modelSelect.append((1, 0))
            modelSelect.append((2, uniAlpha))
            modelSelect.append((3, uniAlpha + biAlpha))
            modelSelect.append((4, uniAlpha + biAlpha + triAlpha))
            modelSelect.append((5, uniAlpha + biAlpha + triAlpha + quadAlpha))

            randNum = random.random()
            selectedModel = 0
            for i in range(0, len(modelSelect) - 1):
                if randNum >= modelSelect[i][1] and randNum < modelSelect[i+1][1]:
                    selectedModel = i+1
            if selectedModel == 0:
                selectedModel = 5

            # Now check if the selected model actual has that history as an ngram. If it does not, then
            # backoff to a lower order model
            hasHistory = False
            ngramProbs = None
            while not hasHistory:
                if selectedModel == 1:
                    hasHistory = True
                else:
                    if selectedModel == 5:
                        ngramProbs = pentagramProbs
                    elif selectedModel == 4:
                        ngramProbs = quadgramProbs
                    elif selectedModel == 3:
                        ngramProbs = trigramProbs
                    else:
                        ngramProbs = bigramProbs
                    for gram in ngramProbs:
                        gramHistory = history[len(history) - (selectedModel - 1):len(history)]
                        gramPrefix = list(gram)[0:selectedModel - 1]
                        if tuple(gramHistory) == tuple(gramPrefix):
                            hasHistory = True
                    # If that history is not contained, then move the model down one order
                    if hasHistory == False:
                        selectedModel -= 1

            # Now, generate a character based on the model
            nextUni = None
            if selectedModel == 1:
                nextUni = unigramGenerator(unigramProbs, history)
            else:
                nextUni = ngramGenerator(selectedModel, ngramProbs, history)

            # Then get the log prob for that unicode character
            logProb = calculateLogLikelihood(nextUni, history, unigramProbs, bigramProbs, trigramProbs, quadgramProbs, pentagramProbs)

            # EXTRA TEST: Tests that the generated characters are in the BMP
            # if is_valid_bmp(nextUni) == False:
            #     print "INVALID GENERATION"
            #     sys.exit()
            # if is_valid_bmp(nextUni):
            #     print "GOOD GENERATION"

            # Write the character and log probability to standard out
            charToPrint = nextUni
            if nextUni == "\n":
                charToPrint = u"NEWLINE"
            sys.stdout.write(charToPrint)
            sys.stdout.write(u"// Generated a character! Probability of generation " + str(logProb))
            sys.stdout.write(u"\n")

            # Then add the character to the history
            del history[0]
            history.append(nextUni)

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

# Calculate the log probability of the given character given the history. Uses all ngram probabilites
def calculateLogLikelihood(nextUni, history, unigramProbs, bigramProbs, trigramProbs, quadgramProbs, pentagramProbs):
    # Now, we calculate the probability for each model
    # We do this by calculating the probability for each ngram model, and then interpolating them
    unigram = nextUni
    prob1 = 0
    if unigram in unigramProbs:
        prob1 = unigramProbs[unigram]
    bigram = (history[3], nextUni)
    prob2 = 0
    if bigram in bigramProbs:
        prob2 = bigramProbs[bigram]
    trigram = (history[2], history[3], nextUni)
    prob3 = 0
    if trigram in trigramProbs:
        prob3 = trigramProbs[trigram]
    quadgram = (history[1], history[2], history[3], nextUni)
    prob4 = 0
    if quadgram in quadgramProbs:
        prob4 = quadgramProbs[quadgram]
    pentagram = (history[0], history[1], history[2], history[3], nextUni)
    prob5 = 0
    if pentagram in pentagramProbs:
        prob5 = pentagramProbs[pentagram]

    # interpolate the probabilites
    totalProb = (uniAlpha * prob1) + (biAlpha * prob2) + (triAlpha * prob3) + (quadAlpha * prob4) + (pentaAlpha * prob5)
    # Take the log base 2 of that probability
    # NOTE: This commented out code is code for testing
    # if (totalProb == 0):
    #     print ("ERROR: A probability is 0")
    #     print ("The offending character is:")
    #     print (nextUni)
    #     print ("This is also the unicode value for it")
    #     print (repr(nextUni))
    #     print ("This is the character as an int")
    #     print (int(nextUni))
    logProb = math.log(totalProb, 2)
    return logProb

# Generates the next character based on a unigram model and the given history
def unigramGenerator(unigramProbs, history):
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

# Generate the next character based on a given ngram model and history
def ngramGenerator(n, ngramProbs, history):
    # Spread the ngram choices between 0 and 1
    ngramLine = []
    firstGram = True
    index = 0
    probSum = 0
    for gram in ngramProbs:
        gramHistory = history[len(history)-(n-1):len(history)]
        gramPrefix = list(gram)[0:n-1]
        if tuple(gramHistory) == tuple(gramPrefix):
            if firstGram:
                ngramLine.append((gram, probSum))
            else:
                ngramLine.append((gram, probSum))
            probSum += ngramProbs[gram]
            index += 1

    # Generate a random number between 0 and 1. Use this number to select a character
    randNum = random.random()
    nextUni = None
    for i in range(0, len(ngramLine) - 1):
        if randNum >= ngramLine[i][1] and randNum < ngramLine[i + 1][1]:
            nextUni = ngramLine[i][0][n-1]
    # If it is equal to None, then it must be the last character
    if nextUni == None:
        nextUni = ngramLine[len(ngramLine) - 1][0][n-1]
    return nextUni

if __name__ == '__main__':
    main(sys.argv[1:])