# -*- coding: utf-8 -*-
from __future__ import division

from collections import Counter
import pickle
import math

import nltk
from nltk.util import ngrams
from nltk.util import pad_sequence

# File path for training data
FILE_PATH = "../dataset_april.txt"
# These are the lines in the file to read 
LINE_START = 0 # inclusive
LINE_END = 2500000 # exclusive


def main():
    print ("Start of program")


    vocab = buildVocabulary()
    print ("This is the length of the vocab: " + str(len(vocab)))

    unigrams = []
    bigrams = []
    trigrams = []
    for uni in vocab:
        unigrams.append(uni)
    totalSymbolCount = len(vocab)


    unigramCount = Counter(unigrams)
    bigramCount = Counter(bigrams)
    trigramCount = Counter(trigrams)
    unigrams = []

    sentences = readLines(FILE_PATH)

    count = 0
    for sentence in sentences:
        count += 1
        if count % 10000 == 0:
            print(count)
            unigramCount = unigramCount + Counter(unigrams)
            bigramCount = bigramCount + Counter(bigrams)
            trigramCount = trigramCount + Counter(trigrams)
            # So I don't run out of memory, reset the lists every 10000
            unigrams = []
            bigrams = []
            trigrams = []
        split = list(sentence)
        split = list(pad_sequence(split, 2, pad_right=True, right_pad_symbol=chr(3)))
        split = list(pad_sequence(split, 3, pad_left=True, left_pad_symbol=chr(2)))

        totalSymbolCount += len(split)
        unigrams += split
        # Purposly ignoring the start symbol right now
        bigrams += list(ngrams(split, 2))
        trigrams += list(ngrams(split, 3))


    unigramCount = unigramCount + Counter(unigrams)
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

    pickle.dump(unigramProb, open("unigram_pickle.p", "wb"))
    pickle.dump(bigramProb, open("bigram_pickle.p", "wb"))
    pickle.dump(trigramProb, open("trigram_pickle.p", "wb"))

    print ("End of program")




### Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    #return [line for line in lines]

    # This will determine the number of lines read from the file
    lines = lines[LINE_START:LINE_END]
    print("number of lines: " + str(len(lines)))
    return lines


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
    with open("../DerivedNames.txt") as f:
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
        print("Vocab size: " + str(len(total_vocab)))
        return total_vocab


if __name__ == '__main__':
    main()