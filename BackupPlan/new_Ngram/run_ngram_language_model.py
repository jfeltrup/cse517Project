# -*- coding: utf-8 -*-

# John Feltrup
# Jamie Park
# Spring Quarter 2018
# CSE 517 Natural Language Processing
#
# A interpolated trigram Model

import sys
import random
import pickle
import math

uniAlpha = 0.4
biAlpha = 0.3
triAlpha = 0.3

def main(argv):
    # Seed the random number generator
    random.seed(argv[0])

    # De-pickle the models
    unigramProbs = pickle.load(open("unigram_pickle.p", "rb"))
    bigramProbs = pickle.load(open("bigram_pickle.p", "rb"))
    trigramProbs = pickle.load(open("trigram_pickle.p", "rb"))

    # Now, set up the history and reading from standard input
    history = [chr(2), chr(2)]

    # Read in the input, make sure to treat it as unicode
    line = sys.stdin.readline()
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
                history = [chr(2), chr(2)]
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

            # Now, we calculate the probability for each model, then interpolate them
            prob1 = unigramProbs[nextUni]
            bigram = (history[1], nextUni)
            prob2 = 0
            if bigram in bigramProbs:
                prob2 = bigramProbs[bigram]
            trigram = (history[0], history[1], nextUni)
            prob3 = 0
            if trigram in trigramProbs:
                prob3 = trigramProbs[trigram]
            # Take the probability to log base 2
            logProb = math.log((uniAlpha * prob1 + biAlpha * prob2 + triAlpha * prob3), 2)

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

             # This generator first tries to generate from the trigram, then the bigram, then the unigram
            check_random = 0.0
            randNum = random.random()
            for trigram in trigramProbs:
                if trigram[0] == history[0] and trigram[1] == history[1]:
                    if check_random + trigramProbs[trigram] > randNum:
                        nextUni = trigram[2]
                        break
                    else:
                        check_random += trigramProbs[trigram]
            if nextUni == None:
                for bigram in bigramProbs:
                    if bigram[0] == history[1]:
                        if check_random + bigramProbs[bigram] > randNum:
                            nextUni = bigram[1]
                            break
                        else:
                            check_random += bigramProbs[bigram]
            if nextUni == None:
                nextUni = unigramGenerator(unigramProbs)

            # Now, we calculate the of that character
            prob1 = unigramProbs[nextUni]
            bigram = (history[1], nextUni)
            prob2 = 0
            if bigram in bigramProbs:
                prob2 = bigramProbs[bigram]
            trigram = (history[0], history[1], nextUni)
            prob3 = 0
            if trigram in trigramProbs:
                prob3 = trigramProbs[trigram]
            logProb = math.log((uniAlpha * prob1 + biAlpha * prob2 + triAlpha * prob3), 2)

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


# Generates the next character based on a unigram model and the given history
def unigramGenerator(unigramProbs):
    check_random = 0.0
    randNum = random.random()
    for gram in unigramProbs:
        if check_random + unigramProbs[gram] > randNum:
            nextUni = gram
            return nextUni
        else:
            check_random += unigramProbs[gram]
    # Shouldn't return in this case
    return chr(0)



if __name__ == '__main__':
    main(sys.argv[1:])