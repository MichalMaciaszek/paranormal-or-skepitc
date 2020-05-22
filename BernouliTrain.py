# coding: utf-8

# In[17]:


def calc_class_logprob(expected_path):
    paranormal_class_count = 0
    skeptic_class_count = 0
    with open(expected_path) as f:
        for line in f:
            if 'P' in line:
                paranormal_class_count += 1
            elif 'S' in line:
                skeptic_class_count += 1
        paranormal_class_prob = paranormal_class_count / (paranormal_class_count + skeptic_class_count)
        skeptic_class_prob = skeptic_class_count / (paranormal_class_count + skeptic_class_count)
    return math.log(paranormal_class_prob), math.log(skeptic_class_prob)


# In[18]:


from collections import defaultdict
import math

# In[19]:


calc_class_logprob('/home/michal/Pobrane/task1/train/expected.tsv')


# In[20]:


def calc_word_prob(in_path, expected_path):
    word_counts = {'paranormal': defaultdict(int), 'skeptic': defaultdict(int)}
    with open(in_path) as in_file, open(expected_path) as exp_file:
        for in_line, exp_line in zip(in_file, exp_file):
            class_ = exp_line.rstrip('\n').replace(' ', '')
            text, timestamp = in_line.rstrip('\n').split('\t')
            tokens = text.lower().split(' ')
            keep_those_words = []
            for token in tokens:
                if token not in keep_those_words:
                    keep_those_words.append(token)
                    if class_ == 'P':
                        word_counts['paranormal'][token] += 1
                    elif class_ == 'S':
                        word_counts['skeptic'][token] += 1
            keep_those_words = []
    return word_counts


# In[21]:


calc_word_prob('/home/michal/Pobrane/task1/train/in.tsv', '/home/michal/Pobrane/task1/train/expected.tsv')


# In[22]:


def calc_word_logprobs(word_counts):
    total_skeptic = sum(word_counts['skeptic'].values()) + len(word_counts['skeptic'].keys())
    total_paranormal = sum(word_counts['paranormal'].values()) + len(word_counts['paranormal'].keys())
    word_logprobs = {'paranormal': {}, 'skeptic': {}}
    for class_ in word_logprobs.keys():
        for token, value in word_counts[class_].items():
            if class_ == 'skeptic':
                word_prob = (value + 1) / total_skeptic
            else:
                word_prob = (value + 1) / total_paranormal
            word_logprobs[class_][token] = math.log(word_prob)
    return word_logprobs


# In[23]:


def bernouli_count_wherenot(word_counts):
    total_skeptic = sum(word_counts['skeptic'].values()) + len(word_counts['skeptic'].keys())
    total_paranormal = sum(word_counts['paranormal'].values()) + len(word_counts['paranormal'].keys())
    word_logprobs = {'paranormal': {}, 'skeptic': {}}
    for class_ in word_logprobs.keys():
        for token, value in word_counts[class_].items():
            if class_ == 'skeptic':
                word_prob = (total_skeptic - value + 1) / total_skeptic
            else:
                word_prob = (total_paranormal - value + 1) / total_paranormal
            word_logprobs[class_][token] = math.log(word_prob)
    return word_logprobs


# In[24]:


paranormal_class_logprob, skeptic_class_logprob = calc_class_logprob('/home/michal/Pobrane/task1/train/expected.tsv')

# In[25]:


word_counts = calc_word_prob('/home/michal/Pobrane/task1/train/in.tsv', '/home/michal/Pobrane/task1/train/expected.tsv')

# In[26]:


wherenotwords = bernouli_count_wherenot(word_counts)

# In[27]:


wherenotwords

# In[14]:


len(word_counts['skeptic'].keys())


# In[38]:


def countemptydict(wherenotwords):
    paranormal_result = 0
    skeptic_result = 0
    word_logprobs = {'paranormal': {}, 'skeptic': {}}
    for class_ in word_logprobs.keys():
        for token, value in wherenotwords[class_].items():
            if class_ == 'skeptic':
                skeptic_result = skeptic_result + value
            else:
                paranormal_result = paranormal_result + value
    return paranormal_result, skeptic_result


# In[41]:


paranormal_result, skeptic_result = countemptydict(wherenotwords)
paranormal_result

# In[15]:


sum(word_counts['skeptic'].values())

# In[16]:


word_logprobs = calc_word_logprobs(word_counts)
word_logprobs
listaparanormal = []
listaskeptic = []
for key, value in word_logprobs['paranormal'].items():
    for keys, values in word_logprobs['skeptic'].items():
        if (key == keys and value > values):
            listaparanormal.append([key,(value-values)])
        elif(key ==keys and values > value):
            listaskeptic.append([key, (values - value)])

print(listaparanormal)
# In[11]:


import pickle

# In[1]:


import pickle
model = (word_counts, paranormal_class_logprob, skeptic_class_logprob, word_logprobs,paranormal_result, skeptic_result,wherenotwords )
with open('./bernouli_model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)

