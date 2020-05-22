# coding: utf-8

# In[1]:


import sys
import pickle

# In[6]:


model = pickle.load(open("./bayes_model.pkl", "rb"))
word_counts, paranormal_class_logprob, skeptic_class_logprob, word_logprobs = model

# In[7]:


word_counts

# In[11]:


word_logprobs




from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')

# In[12]:


for line in sys.stdin:
    document = line.rstrip()
    terms = tokenizer.tokenize(document)
    for term in terms:
        if term not in word_counts['skeptic'].items():
            word_counts['skeptic'][term] = 0
        if term not in word_counts['paranormal'].items():
            word_counts['paranormal'][term] = 0

        # log_prob_sceptic += math.log((sceptic_words[term] + 1) / (sceptic_words_total + vocabulary_size))
        # log_prob_paranormal += math.log((paranormal_words[term] + 1) / (paranormal_words_total + vocabulary_size))

    if skeptic_class_logprob > paranormal_class_logprob:
        print(" S")
    else:
        print(" P")
