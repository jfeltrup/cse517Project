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
    #testUnicodeFunction()

    print "End of program"

# Takes the training data, and makes a list of unicode sentences
# Returns a list of unicode strings
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
    # Set up the vocabulary with all unicode characters
    vocabulary = []
    for i in range(0, 65424):
        vocabulary.append(unichr(i))
    #print vocabulary

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
    totalSymbolCount = 65424

    # This is all test unicode sentences
    # test1 = u"碼統統統統統cat"
    # test2 = u"ich sih in grâwen tägelîch als er wil tagen"
    # test3 = u"cat in the hat"
    # sentences = [test1, test2, test3]

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

        # Another way of padding the sentences, based on the individual model
        #unigrams += list(ngrams(split, 1))
        # unigrams += split
        # bigrams += list(ngrams(split, 2, pad_left=True, pad_right=True, left_pad_symbol=unichr(2), right_pad_symbol=unichr(3)))
        # trigrams += list(ngrams(split, 3, pad_left=True, pad_right=True, left_pad_symbol=unichr(2), right_pad_symbol=unichr(3)))
        # quadgrams += list(ngrams(split, 4, pad_left=True, pad_right=True, left_pad_symbol=unichr(2), right_pad_symbol=unichr(3)))
        # pentagrams += list(ngrams(split, 5, pad_left=True, pad_right=True, left_pad_symbol=unichr(2), right_pad_symbol=unichr(3)))


    # print Counter(unigrams)
    # print Counter(bigrams)
    # print Counter(trigrams)
    # print Counter(quadgrams)
    # print Counter(pentagrams)

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

    # print unigramProb
    # print bigramProb
    # print trigramProb
    # print quadgramProb
    # print pentagramProb

    # Now pickle the models to use for later
    # Comment this out while tuning parameters
    pickle.dump(unigramProb, open("unigram_pickle.p", "wb"))
    pickle.dump(bigramProb, open("bigram_pickle.p", "wb"))
    pickle.dump(trigramProb, open("trigram_pickle.p", "wb"))
    pickle.dump(quadgramProb, open("quadgram_pickle.p", "wb"))
    pickle.dump(pentagramProb, open("pentagram_pickle.p", "wb"))

    # This is for testing and tuning the model
    #tuneModel(unigramProb, bigramProb, trigramProb, quadgramProb, pentagramProb)

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
    # To do that, I will need to take the sum of all log probabilites, divide it by the number of
    # words in the corpus, then to 2 to the negative of that sum

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

    # Great, now I just need to check if this works a bit later, tune a bit, wrap it back up,
    # Then I can call it good unless I want to improve it some more.

# Function for caclulating the log probability based on the data
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



# This is a lot of code in testing how to work with unicode characters and make sure they are
# process properly
def testUnicodeFunction():
    normal = "The cat in the hat"

    # First things first, lets see how to split up unicode strings
    test = u"碼統統統統統cat"


    # testing encoding and decoding
    test2 = "cat"
    test3 = test2.encode("utf-8")
    print test3
    print list(test3)
    test4 = test3.decode("utf-8")
    print test4

    #items = test.decode("utf-8")

    # How can I get it to print in unicode land?
    print test.encode('utf-8')

    split = list(test)

    # Okay, so doing list(string) will split up the unicode characters. good to know
    print "Testing splitting unicode characters"
    print split
    print split[0]
    print split[1]

    print "testing adding to unicde"
    print split[0]
    print split[0] + unichr(1)
    # Got it. This is how I will populate the the vocabulary
    print unichr(1)
    print unichr(10)
    print unichr(100)
    # I guess I can't entirely view it as I expected, but screw it, this can work
    print unichr(10000).encode("utf-8")

    print "testing converting to unichr"
    # Now I will test converting the results to unichr
    # new_list = []
    # for thing in split:
    #     new_list.append(unichr(thing))
    # print new_list

    # now some nltk testing
    # Of course it's broken. Not sure why I expected anything else

    # Okay, the tokenize is broken/non-existent. Good to know
    print "Now some nltk testing"
    b = nltk.word_tokenize(normal)
    print b


    padded = list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

    a = ngrams(split, 2)

    print list(a)
    print Counter(a)


if __name__ == '__main__':
    main()