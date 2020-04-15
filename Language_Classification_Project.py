#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries
# 

# In[ ]:


import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import string

from collections import defaultdict

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import joblib
import pickle as pkl

from helper_code import *


# ## Data Exploration and Visualization

# In[ ]:


#All data is saved under the name train_sentences.(language abbreviation) or val_sentences.(language abbreviation)
#The data is first parsed and read using readlines and then the data is passed onto the keyvalue in data_raw
#The encoding might differ but it can be solved easily, I used utf8 as well as cp1252

def open_file(filename, encoding):
    with open(filename, 'r',encoding='utf8') as f:
        data = f.readlines()
    return data


# In[ ]:


data_raw=dict ()

data_raw['sk']=open_file('Data/Sentences/train_sentences.sk', encoding="utf8")
data_raw['cs']=open_file('Data/Sentences/train_sentences.cs', encoding="utf8")
data_raw['en']=open_file('Data/Sentences/train_sentences.en', encoding="utf8")

#Here the encoding changed, so iused the open function and directly apploed readlines
data_raw['fr']= open('Data/Sentences/train_sentences.fr',encoding='cp1252').readlines()


# In[ ]:


#this function is used to show the statistics of our data set for each language
def show_statistics(data):
    for language, sentences in data.items():
        #the values saved inside each language such as sk, cs and en are then split using whitespace
        word_list = ' '.join(sentences).split()
        number_of_sentences = len(sentences)
        number_of_words = len(word_list)
        #unique words list is calculated using sets as sets dont repeat their contents
        number_of_unique_words = len(set(word_list))
        #sample extract , just to show what is present in the data as we move forward
        sample_extract = ''.join(sentences[0].split()[0:7])
        
       
        print(f'Language: {language}')
        print('-----------------------')
        print(f'Number of sentences\t:\t {number_of_sentences}')
        print(f'Number of words\t\t:\t {number_of_words}')
        print(f'Number of unique words\t:\t {number_of_unique_words}')
        print(f'Sample extract\t\t:\t {sample_extract}...\n')


# In[ ]:


show_statistics(data_raw)


# In[ ]:


def preprocess(text):
    '''
    Removes punctuation and digits from a string, and converts all characters to lowercase. 
    Also clears all \n and hyphens (splits hyphenated words into two words).
    
    '''
    
    preprocessed_text = text
    #.replace is used extensively to remove unwanted chars in the data
    preprocessed_text=text.lower().replace('-',' ')
    #import string done for string.punctuation
    translation_table=str.maketrans('\n',' ',string.punctuation+string.digits)
    preprocessed_text=preprocessed_text.translate(translation_table)
    
    return preprocessed_text


# In[ ]:


#function is run and saved as dictionary 
data_preprocessed={k:[preprocess(sentence)for sentence in v] for k,v in data_raw.items()}


# ## The Naive Bayes Model
# 

# **Bayes' Theorem**
# 
# \begin{equation}
# P(A | B)=\frac{P(B | A) \times P(A)}{P(B)}
# \end{equation}
# 
# Now, let's translate this theory into our specific problem. In our case, where we want to categorise a sentence `my name is Abhi` into one of `sk`, `cs`, or `en`, the following are the probabilities we want to determine.
# 
# \begin{equation}
# P(\text {sk} | \text {my name is Abhi})=\frac{P(\text {my name is Abhi} | \text {sk}) \times P(\text {sk})}{P(\text {my name is Abhi})}
# \end{equation}
# 
# \begin{equation}
# P(\text {cs} | \text {my name is Abhi})=\frac{P(\text {my name is Abhi} | \text {cs}) \times P(\text {cs})}{P(\text {my name is Abhi})}
# \end{equation}
# 
# \begin{equation}
# P(\text {en} | \text {my name is Abhi})=\frac{P(\text {my name is Abhi} | \text {en}) \times P(\text {en})}{P(\text {my name is Abhi})}
# \end{equation}

# ## Unseen Data
# 
# Since we assume conditional independence across our features, our numerator term for any of the above equations can be broken into the following.
# 
# \begin{equation}
# P(\text {my name is Abhi} | \text {en}) = P(\text {my} | \text {en}) \times P(\text {name} | \text {en}) \times P(\text {is} | \text {en}) \times P(\text {Abhi} | \text {en})
# \end{equation}

# ## Vectorizing Training Data

# |Sentence   	||   my   	| is 	| I 	| love 	| name 	| it 	| Abhi 	|
# |-----------------	||:------:	|:--:	|:-:	|:----:	|:----:	|:--------:	|:---:	|
# | my name is Abhi  	||    1   	|  1 	| 0 	|   0  	|   1  	|     0    	|  1  	|
# | I love it 	||    0   	|  0 	| 1 	|   1  	|   0  	|     1    	|  0  	|

# In[ ]:


#Each key/value is saved in different lists from the data_preprocessed dictionary by iterating through the dict.
sentences_train, y_train=[],[]
for k,v in data_preprocessed.items():
    for sentence in v:
        sentences_train.append(sentence)
        y_train.append(k)
        


# In[ ]:


'''By using CountVectorizer function we can convert text document to
matrix of word count. Matrix which is produced here is sparse matrix.'''

vectorizer= CountVectorizer()


# In[ ]:


#Basically whats happening here is that the y_train contains the words that are appearing in the dict
#while at the same time, using vectorizer, the number of times the words are repeating is the x train.
x_train=vectorizer.fit_transform(sentences_train)


# ## Initializing Model Parameters and Training

# In[ ]:


#we use the MultinomialNB from the bayes naive module.
#multinomial naive bayes explicitly models the word counts 
#and adjusts the underlying calculations to deal with in.
naive_classifier=MultinomialNB()
naive_classifier.fit(x_train,y_train)


# ## Vectorizing Validation Data and Evaluating Model

# In[ ]:


'''This is basically to store the data of each language in a dictionary, while maintaining the fact that
the encoding is kept in mind, while at the same time, data_val is used with larger datasets,
just so that the validation has much more words to vector upon.
'''
data_val=dict()
data_val['sk']=open_file('Data/Sentences/val_sentences.sk', encoding='utf8')
data_val['cs']=open_file('Data/Sentences/val_sentences.cs', encoding='utf8')
data_val['en']=open_file('Data/Sentences/val_sentences.en', encoding='utf8')
data_val['fr']=open('Data/Sentences/val_sentences.fr', encoding='cp1252').readlines()

data_val_preprocessed={k:[preprocess(sentence)for sentence in v] for k,v in data_val.items()}


# In[ ]:


#this is the same process as training model
sentences_val, y_val=[],[]
for k,v in data_preprocessed.items():
    for sentence in v:
        sentences_val.append(sentence)
        y_val.append(k)
        
        


# In[ ]:


'''
In scikit-learn estimator api,

fit() : used for generating learning model parameters from training data
transform() : parameters generated from fit() method,applied upon model to generate transformed data set.
fit_transform() : combination of fit() and transform() api on same data set

'''
x_val=vectorizer.transform(sentences_val)


# In[ ]:


'''
the predict functionality is used to make up a prediction about the dataset using 
the naive bayes thm
'''
predictions= naive_classifier.predict(x_val)


# In[ ]:


plot_confusion_matrix(y_val,predictions,['sk','cs','en','fr'])


# In[ ]:


f1_score(y_val,predictions,average='weighted')


# ## Simple Adjustments and Highlighting Model Shortcomings

# In[ ]:


"""
The predictions can be improved by playing with the alpha value.
In Multinomial Naive Bayes, the alpha parameter is what is known as a hyperparameter; 
i.e. a parameter that controls the form of the model itself. 
In most cases, the best way to determine optimal values for hyperparameters is through
a grid search over possible parameter values, using cross validation to evaluate the performance
of the model on your data at each value.


"""
naive_classifier=MultinomialNB(alpha=1.0,fit_prior=False)
naive_classifier.fit(x_train,y_train)
predictions=naive_classifier.predict(x_val)
plot_confusion_matrix(y_val,predictions,['sk','cs','en','fr'])


# In[ ]:


f1_score(y_val,predictions,average='weighted')


# ## Using Subwords to Shift Perspective

# **Dummy Dataset**
# 
# playing ; eating ; play ; reads ; tea
# 
# **Step 1**
# 
# Break each word into characters
# 
# playing > p l a y i n g
# 
# 
# **Step 2**
# 
# Find common character sequences
# 
# ea, ing, play
# 
# **Step 3**
# 
# Convert dataset using these subwords into
# 
# play ing ; ea t ing ; play ; r ea d s ; t ea

# ### The subwords model is much more accurate than conventional classification

# ### So we improvise and run the same lines of code for the subwords model
# ### and implement our data set to train through Byte Pair Encoding

# In[ ]:


# taken from https://arxiv.org/abs/1508.07909
# to understand more about subwords in NLP : https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46
#its BPE: Byte Pair Encoding

import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int) 
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq 
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word] 
    return v_out


# In[ ]:


def get_vocab(data):

    words = []
    for sentence in data:
        words.extend(sentence.split())
        
    vocab = defaultdict(int)
    for word in words:
        vocab[' '.join(word)] += 1
        
    return vocab


# In[ ]:


vocab = get_vocab(sentences_train)


# In[ ]:


# also taken from original paper
for i in range(100):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get) 
    vocab = merge_vocab(best, vocab)


# In[ ]:


merges = defaultdict(int)
for k, v in vocab.items():
    for subword in k.split():
        if len(subword) >= 2:
            merges[subword] += v


# In[ ]:


merge_ordered = sorted(merges, key=merges.get, reverse=True)


# In[ ]:


pkl.dump(merge_ordered, open('Data/Auxiliary/merge_ordered.pkl', 'wb'))


# In[ ]:


def split_into_subwords(text):
    merges = pkl.load(open('Data/Auxiliary/merge_ordered.pkl', 'rb'))
    subwords = []
    for word in text.split():
        for subword in merges:
            subword_count = word.count(subword)
            if subword_count > 0:
                word = word.replace(subword, ' ')
                subwords.extend([subword]*subword_count)
    return ' '.join(subwords)


# In[ ]:


split_into_subwords('hello my name is abhishek')


# In[ ]:


data_preprocessed_subwords={k:[split_into_subwords(sentence)for sentence in v] for k,v in data_preprocessed.items()}


# In[ ]:


data_train_subwords=[]
for sentence in sentences_train:
    data_train_subwords.append(split_into_subwords(sentence))


# In[ ]:


data_val_subwords=[]
for sentence in sentences_val:
    data_val_subwords.append(split_into_subwords(sentence))


# In[ ]:


vectorizer= CountVectorizer()


# In[ ]:


x_train=vectorizer.fit_transform(data_train_subwords)
x_val=vectorizer.transform(data_train_subwords)


# In[ ]:


naive_classifier=MultinomialNB(alpha=1,fit_prior=False)


# In[ ]:


naive_classifier.fit(x_train,y_train)
predictions=naive_classifier.predict(x_val)
plot_confusion_matrix(y_val,predictions,['sk','cs','en','fr'])


# In[ ]:


f1_score(y_val,predictions,average='weighted')


# ### Final step is to save these models as joblib files in respective folders to access it when we want

# In[ ]:


joblib.dump(naive_classifier,'Data/Models/final_model.joblib')


# In[ ]:


joblib.dump(vectorizer,'Data/Vectorizers/final_model.joblib')


# ### Now, we test our code by calling the model

# In[ ]:


model=joblib.load('Data/Models/final_model.joblib')
vectorizer = joblib.load('Data/Vectorizers/final_model.joblib')


# In[ ]:



text='Hi I am Abhishek'
text=preprocess_function(text)
text=[split_into_subwords_function(text)]
text_vectorized= vectorizer.transform(text)

model.predict(text_vectorized)

