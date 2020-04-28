# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:23:12 2018

@author: kulpatil
"""
import re, pandas as pd
import string, nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
en_stop = set(stopwords.words('english'))
#en_stop.update(['ml','m','shampoo']) # remove it if you need punctuation 


frequency = {}
import csv
from nltk.corpus import wordnet as wn
training_set = []
from collections import Counter
cnt = Counter()
cnt2= Counter()
cnt3= Counter()
# create sample documents
with open('Care_Products.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_set.append((row['post'])) 
# loop through document list
texts = []
for i in training_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)  
    #tokens=[word for word in tokens if not word in stopwords.words()]
    words = [word for word in tokens if word.isalpha()]
    # remove stop words from tokens
    stopped_tokens = [i for i in words if (not i in en_stop and len(str(i)) > 2 )]
    texts.append(stopped_tokens)
    for word in stopped_tokens:
        cnt[word] += 1 
    #Create your bigrams
    bgs = nltk.bigrams(stopped_tokens)
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(bgs)
    for k,v in fdist.items():
        cnt2[k] += 1

    fdist = nltk.FreqDist(nltk.trigrams(stopped_tokens))
    for k,v in fdist.items():
        cnt3[k] += 1
        
freq=cnt+cnt2+cnt3

complete_data=[]
for value, count in freq.most_common():
    complete_data.append([value,count])
  
writer = pd.DataFrame(complete_data, columns=['Keywords', 'Frequency'])    
writer.to_csv("FrequencyUniBiGram_Care_Products.csv", index = None, header=True) 
