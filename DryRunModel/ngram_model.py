# -*- coding: utf-8 -*-

# John Feltrup
# Spring Quarter 2018
# CSE 517 Natural Language Processing

# This program was used to generate the ngram models, as well as do some testing
# Question: Should I turn this one in? Maybe ask on monday

from __future__ import division

import nltk
from nltk.util import ngrams
from nltk.util import pad_sequence
from collections import Counter
import pickle
import math

import unicodedata

import pandas as pd

# Hyperparameters to test for interpolating the model
uniAlpha = 0.2
biAlpha = 0.1
triAlpha = 0.1
quadAlpha = 0.3
pentaAlpha = 0.3

def main():
    print "Start of program"

    createModel()

    # Use this function for testing the Unicode 10 vocabulary
    #testAllUnicode10()

    print "End of program"

# TODO: Change this function to read from the project Database
# Reads in the training Data from a file, and converts it into a list of unicode sentences
# This list of sentences is returned
def processData():
    # This is reading from the data.
    data = pd.read_table("umass_global_english_tweets-v1/all_annotated.tsv")

    # This is the row count to use for the model
    rowCount = data.shape[0]
    # This is the row count to use for tuning the parameters
    #rowCount = 9452

    sentences = []
    for i in range(0, rowCount):
        sentences.append(unicode(data.iloc[i][3], "utf-8"))

    # Then return the sentences
    return sentences

# Builds the ngram models based on the training data, and save the 1-5 grams in python pickle files
def createModel():
    # To see how vocabulary is built, see buildVocabulary function
    vocabulary = buildVocabulary()
    # Init the 1-5 grams
    unigrams = []
    bigrams = []
    trigrams = []
    quadgrams = []
    pentagrams = []

    # So no unicode characters in the Plane are unknown, init their counts to
    # 1 for each character
    for uni in vocabulary:
        unigrams.append(uni)
    totalSymbolCount = len(vocabulary)

    # Now this is code that used the training data
    sentences = processData()

    for sentence in sentences:
        # Split the sentence, then give a different padding for every length
        split = list(sentence)
        totalSymbolCount += len(split)

        # Consistent padded version
        split = list(pad_sequence(split, 2, pad_right=True, right_pad_symbol=unichr(3)))
        split = list(pad_sequence(split, 5, pad_left=True, left_pad_symbol=unichr(2)))

        unigrams += split
        bigrams += list(ngrams(split, 2))
        trigrams += list(ngrams(split, 3))
        quadgrams += list(ngrams(split, 4))
        pentagrams += list(ngrams(split, 5))

    # Get the counts for each type of gram
    unigramCount = Counter(unigrams)
    bigramCount = Counter(bigrams)
    trigramCount = Counter(trigrams)
    quadgramCount = Counter(quadgrams)
    pentagramCount = Counter(pentagrams)

    # Calculate the probability for each type of gram
    unigramProb = {}
    for gram in unigramCount:
        unigramProb[gram] = unigramCount[gram] / totalSymbolCount

    bigramProb = {}
    for gram in bigramCount:
        lastGram = gram[0]
        bigramProb[gram] = bigramCount[gram] / unigramCount[lastGram]

    trigramProb = {}
    for gram in trigramCount:
        lastGram = (gram[0], gram[1])
        trigramProb[gram] = trigramCount[gram] / bigramCount[lastGram]

    quadgramProb = {}
    for gram in quadgramCount:
        lastGram = (gram[0], gram[1], gram[2])
        quadgramProb[gram] = quadgramCount[gram] / trigramCount[lastGram]

    pentagramProb = {}
    for gram in pentagramCount:
        lastGram = (gram[0], gram[1], gram[2], gram[3])
        pentagramProb[gram] = pentagramCount[gram] / quadgramCount[lastGram]

    # Now pickle the models to use for later
    # Comment this out while tuning parameters
    pickle.dump(unigramProb, open("unigram_pickle.p", "wb"))
    pickle.dump(bigramProb, open("bigram_pickle.p", "wb"))
    pickle.dump(trigramProb, open("trigram_pickle.p", "wb"))
    pickle.dump(quadgramProb, open("quadgram_pickle.p", "wb"))
    pickle.dump(pentagramProb, open("pentagram_pickle.p", "wb"))

    # This is for testing and tuning the model
    #tuneModel(unigramProb, bigramProb, trigramProb, quadgramProb, pentagramProb)


# Returns a list of all the unicode characters that are a part of our valid vocabulary
# The characters are read from the blocks in Blocks.txt
# Lines in Blocks.txt that start with a # will be ignored
def buildVocabulary():
    total_vocab = []
    # Now read the file
    with open("Blocks.txt") as f:
        line = f.readline()
        while line:
            if line.startswith("#") or line.startswith("\n"):
                line = f.readline()
            else:
                print line
                split1 = line.split(";")
                print split1
                split2 = split1[0].split("..")
                print split2

                lower = int(split2[0], 16)
                upper = int(split2[1], 16)

                for i in range(lower, upper+1):
                    total_vocab.append(i)

                line = f.readline()
        return total_vocab

# NOTE: This function is not set up for the Project Dataset. It will not work until modified
# TODO: replace the dev set in this function with the Project Dataset. Will not work until change is made
# This function is used to calculate the perplexity of the model on a held out Dev Set
def tuneModel(unigramProbs, bigramProbs, trigramProbs, quadgramProbs, pentagramProbs):
    print "In tuning model function"

    # Training set 0 - 9,452
    # Dev set 9,452 - rowCount
    # Read the sentences that are for the dev set
    data = pd.read_table("umass_global_english_tweets-v1/all_annotated.tsv")
    rowCount = data.shape[0]
    devSentences = []
    for i in range(9452, rowCount):
        devSentences.append(unicode(data.iloc[i][3], "utf-8"))
    print "length of dev sentences"
    print len(devSentences)

    # Now calculate the perplexity of the dev set
    # First, the sum of the log probabilites
    totalSymbols = 0.0
    sumOfLog = 0.0
    history = [unichr(2), unichr(2), unichr(2), unichr(2)]

    for sentence in devSentences:
        split = list(sentence)
        for i in range(0, len(split)):
            nextUni = split[i]
            totalSymbols += 1
            sumOfLog += calculateLogLikelihood(nextUni, history, unigramProbs, bigramProbs, trigramProbs, quadgramProbs, pentagramProbs)

    # After we have gone through all the sentence, calculate the rest of the probability
    termL = sumOfLog / totalSymbols
    perplexity = math.pow(2, -termL)

    print "This is the perplexity"
    print perplexity

# Function for caclulating the log probability based on the data. Takes a unicode character, character history,
# and all 5 ngram models
def calculateLogLikelihood(nextUni, history, unigramProbs, bigramProbs, trigramProbs, quadgramProbs,
                           pentagramProbs):
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
    totalProb = (uniAlpha * prob1) + (biAlpha * prob2) + (triAlpha * prob3) + (quadAlpha * prob4) + (
    pentaAlpha * prob5)
    # Take the log base 2 of that probability
    logProb = math.log(totalProb, 2)
    return logProb

# A function to test reading in the Unicode 10 characters. Only used for testing, not for building the Model
def testAllUnicode10():
    print "starting all unicode 10 test"

    total_vocab = []

    # Now read the file
    with open("Blocks.txt") as f:
        line = f.readline()
        while line:
            if line.startswith("#") or line.startswith("\n"):
                line = f.readline()
            else:
                print line
                split1 = line.split(";")
                print split1
                split2 = split1[0].split("..")
                print split2

                lower = int(split2[0], 16)
                upper = int(split2[1], 16)

                print "lower"
                print hex(lower)
                print "upper"
                print hex(upper)
                print ()

                #if lower < int("FFFF", 16):
                    # now lets try and get characters in that range
                for i in range(lower, upper+1):
                    total_vocab.append(i)

                line = f.readline()

        print len(total_vocab)
            # # scrap off the first character
            # seperated = line.split(";")
            # hex_string = seperated[0]
            #
            # print hex_string
            #
            # hex_str = hex_string
            # hex_int = int(hex_str, 16)
            # uni = unichr(hex_int)
            # print hex(hex_int)
            # #print uni.encode('utf-8')
            #
            # print()
            #
            # line = f.readline()


if __name__ == '__main__':
    main()