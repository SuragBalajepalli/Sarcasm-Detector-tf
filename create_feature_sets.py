3"""
This file is responsible to take the raw sentences as input and then extract
features from them.
This file needs two files with positive and regular data that is cleaned and
pre-processed. The names of the files should be 'negproc.npy' and 'posproc.npy'

"""

import numpy as np
from textblob import TextBlob
import nltk
import string
import exp_replace
import random

# Read the data from numpy files into arrays
sarcastic_data = np.load('posproc.npy')
regular_data = np.load('negproc.npy')

featuresets = []
classes = ["SARCASTIC", "REGULAR"]


def extractFeatures():
    print("We have " + str(len(sarcastic_data)) + " Sarcastic sentences.")
    print("We have " + str(len(regular_data)) + " Regular sentences.")

    print("Extracting features for negative set")
    # We have 4 times more Regular data as Positive data. Hence we only take
    # every 4th sentence from the Regular data.
    for x in regular_data[::4]:
        features = extractFeatureOfASentence(x)
        featuresets.append([features, [0, 1]])

    print("Extracting features for positive set")
    for x in sarcastic_data:
        features = extractFeatureOfASentence(x)
        featuresets.append([features, [1, 0]])

    # Shuffle the featuresets so that thy are not in any paticular order
    random.shuffle(featuresets)
    featuresets1 = np.array(featuresets)

    # Save the features into a numpy file.
    np.save('featuresets', featuresets1)


def extractFeatureOfASentence(sen):
    features = []

    # Tokenize the sentence and then convert everthing to lower case.
    tokens = nltk.word_tokenize(exp_replace.replace_emo(str(sen)))
    tokens = [(t.lower()) for t in tokens]

    # Extract features of full sentence.
    fullBlob = TextBlob(joinTokens(tokens))
    features.append(fullBlob.sentiment.polarity)
    features.append(fullBlob.sentiment.subjectivity)

    # Extract features of halves.
    size = len(tokens) // 2
    parts = []
    i = 0
    while i <= len(tokens):
        if i == size:
            parts.append(tokens[i:])
            break
        else:
            parts.append(tokens[i:i + size])
            i += size
    for x in range(0, len(parts)):
        part = parts[x]
        halfBlob = TextBlob(joinTokens(part))
        features.append(halfBlob.sentiment.polarity)
        features.append(halfBlob.sentiment.subjectivity)
    features.append(np.abs(features[-2] - features[-4]))

    # Extract features of thirds.
    size = len(tokens) // 3
    parts = []
    i = 0
    while i <= len(tokens):
        if i == 2 * size:
            parts.append(tokens[i:])
            break
        else:
            parts.append(tokens[i:i + size])
            i += size

    ma = -2
    mi = 2
    for x in range(0, len(parts)):
        part = parts[x]
        thirdsBlob = TextBlob(joinTokens(part))
        pol = thirdsBlob.sentiment.polarity
        sub = thirdsBlob.sentiment.subjectivity
        if pol > ma:
            ma = pol
        if pol < mi:
            mi = pol
        features.append(pol)
        features.append(sub)
    features.append(np.abs(ma - mi))

    # Extract features of fourths.
    size = len(tokens) // 4
    parts = []
    i = 0
    while i <= len(tokens):
        if i == 3 * size:
            parts.append(tokens[i:])
            break
        else:
            parts.append(tokens[i:i + size])
            i += size
    ma = -2
    mi = 2
    for x in range(0, len(parts)):
        part = parts[x]
        fourthsBlob = TextBlob(joinTokens(part))
        pol = fourthsBlob.sentiment.polarity
        sub = fourthsBlob.sentiment.subjectivity
        if pol > ma:
            ma = pol
        if pol < mi:
            mi = pol
        features.append(pol)
        features.append(sub)
    features.append(np.abs(ma - mi))

    return features


def joinTokens(t):
    s = ""
    for i in t:
        if i not in string.punctuation and not i.startswith("'"):
            s += (" " + i)
    return s.strip()


if __name__ == '__main__':
    extractFeatures()
