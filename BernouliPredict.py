# coding: utf-8

# In[2]:


import sys
import re
import pandas as pd
from collections import Counter
import pickle

model = pickle.load(open("./bernouli_model.pkl", "rb"))
word_counts, paranormal_class_logprob, skeptic_class_logprob, word_logprobs ,paranormal_result, skeptic_result,wherenotwords = model

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')

# In[7]:


for line in sys.stdin:
    document = line.rstrip()
    terms = tokenizer.tokenize(document)
    wholelogparanomal = paranormal_class_logprob + paranormal_result
    wholelogskeptic = skeptic_class_logprob + skeptic_result
    for term in terms:
        if term in word_logprobs['skeptic'].keys():
            wholelogskeptic = wholelogskeptic + word_logprobs['skeptic'][term]
        if term in wherenotwords['skeptic'].keys():
            wholelogskeptic = wholelogskeptic - wherenotwords['skeptic'][term]
        if term in word_logprobs['paranormal'].keys():
            wholelogparanomal = wholelogparanomal + word_logprobs['paranormal'][term]
        if term in wherenotwords['paranormal'].keys():
            wholelogskeptic = wholelogskeptic - wherenotwords['paranormal'][term]

    if wholelogskeptic > wholelogparanomal:
        print(" S")
    else:
        print(" P")

