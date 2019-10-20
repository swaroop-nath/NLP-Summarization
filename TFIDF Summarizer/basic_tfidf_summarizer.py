from bs4 import BeautifulSoup as BS
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
from numpy import log

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

# Calculating the frequency of words
term_freq_matrix = {}

for word in words:
    if term_freq_matrix.get(word) == None:
        term_freq_matrix[word] = 1
    else: term_freq_matrix[word] += 1
    
# Standardizing the frequency
count_words = len(term_freq_matrix)
for key, value in term_freq_matrix.items():
    term_freq_matrix[key] = value/count_words
    
# Calculating Inverse Document Frequency - treating each sentence as a document
num_docs = len(original_sentences)

idf_matrix = {}

for word in words:
    if idf_matrix.get(word) == None:
        count = 0
        for sent in sentences:
            if word in sent: count += 1
        idf_matrix[word] = count

for key in idf_matrix.keys():
    idf_matrix[key] = log(num_docs/idf_matrix[key])
   
# Calculating the TFIDF score for each word
tf_idf_score = {}

for key in term_freq_matrix.keys():
    tf_idf_score[key] = term_freq_matrix[key] * idf_matrix[key]
    
# Scoring sentences
sentence_scores = {}

for sent in original_sentences:
    sentence_scores[sent] = 0
    
for i, sent in enumerate(sentences):
    sent_words = word_tokenize(sent)
    for word in sent_words:
        if word in words:
            sentence_scores[original_sentences[i]] += tf_idf_score[word]
            
num_output_sentences = 4
threshold = sorted(list(sentence_scores.values()), reverse = True)[num_output_sentences]

summary = []

for sent in original_sentences:
    if sentence_scores[sent] > threshold:
        summary.append(sent)
      
summary = ' '.join(summary)

with open('output.txt', 'w') as file:
    file.write(summary)