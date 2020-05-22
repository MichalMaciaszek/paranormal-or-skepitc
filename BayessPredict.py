#!/usr/bin/python3

import sys
import pickle
from math import log
from tokenizator import tokenize

model = pickle.load(open('model.pkl', 'rb'))
pskeptic, vocabulary_size, skeptic_words_total, paranormal_words_total, skeptic_count, paranormal_count = model

for line in sys.stdin:
    document = line.strip().split('\t')[0]
    terms = tokenize(document)

    log_prob_skeptic = log(pskeptic)
    log_prob_paranormal = log(1 - pskeptic)

    for term in terms:
        if term not in skeptic_count:
            skeptic_count[term] = 0
        if term not in paranormal_count:
            paranormal_count[term] = 0

        log_prob_skeptic += log((skeptic_count[term] + 1) / (skeptic_words_total + vocabulary_size))
        log_prob_paranormal += log((paranormal_count[term] + 1) / (paranormal_words_total + vocabulary_size))

    if log_prob_skeptic > log_prob_paranormal:
        print(0.15)
    else:
        print(0.85)