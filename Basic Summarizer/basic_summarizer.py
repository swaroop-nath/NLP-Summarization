import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords

# Importing the dataset
raw_data = ''
with open('../data.txt', 'rb') as file:
    raw_data = file.read()

# Cleaning the data
data = BS(raw_data).text

non_space_regex = '[0-9\[\]/,()–"\']' # Removing numbers, square and round brackets, quotes, apostrophes, hypens, obliques
data_clean = re.sub(non_space_regex, '', data)

sentences = sent_tokenize(data_clean)
original_sentences = sent_tokenize(data)

# Removing punctuations in each sentence (like periods)
sentences = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in sentences]

words = [word for sent in sentences for word in word_tokenize(sent)]

# Removing stopwords from words list as well as sentences list
stop_words = stopwords.words('english')
words = [word for word in words if word not in stop_words]

term_freq_matrix = {}

# Calculating the frequency of words
for word in words:
    if term_freq_matrix.get(word) == None:
        term_freq_matrix[word] = 1
    else: term_freq_matrix[word] += 1
    
# Standardizing the frequency
count_words = len(term_freq_matrix)
for key, value in term_freq_matrix.items():
    term_freq_matrix[key] = value/count_words
    
# Scoring sentences
sentence_scores = {}

for sent in original_sentences:
    sentence_scores[sent] = 0
    
for i, sent in enumerate(sentences):
    sent_words = word_tokenize(sent)
    for word in sent_words:
        if word in words:
            sentence_scores[original_sentences[i]] += term_freq_matrix[word]
            
num_output_sentences = 4
threshold = sorted(list(sentence_scores.values()))[num_output_sentences + 1]

summary = []

for sent in original_sentences:
    if sentence_scores[sent] > threshold:
        summary.append(sent)
        
summary = ' '.join(summary)

with open('output.txt', 'w') as file:
    file.write(summary)