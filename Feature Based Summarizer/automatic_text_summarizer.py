from tf_idf_vectorizer import tf_idf_vectorize
import pandas as pd
from bs4 import BeautifulSoup as BS
from nltk import sent_tokenize, word_tokenize
import numpy as np

#Reading data from the file
raw_data = ''
with open('../data.txt', 'r') as source:
    raw_data = source.read()
    
data = BS(raw_data).text

#Obtaining the tf/tf-idf score for the sentences
sentences_with_scores = tf_idf_vectorize(data)

columns = ['tf-score', 'length-score', 'position-score', 'paragraph-score', 'cue-words-score']

sentences = sent_tokenize(data)

#Storing all the data in a DataFrame
vector_space = pd.DataFrame(index = sentences, columns = columns)

for sentence, tf_idf_score in sentences_with_scores.items():
    vector_space.loc[sentence, 'tf-score'] = tf_idf_score
    
#Using Barrera and Verma's first model to score sentence based on the position
total_sentences = len(sentences)
alpha = 2
    
for index, sentence in enumerate(sentences):
    vector_space.loc[sentence, 'position-score'] = (np.cos((2*np.pi*index)/(total_sentences - 1)) + alpha - 1)/(alpha)
    vector_space.loc[sentence, 'length-score'] = len(word_tokenize(sentence))

mean = np.mean(vector_space['length-score'])
std_dev = np.sqrt(np.var(vector_space['length-score']))
max_val = max(np.abs(min(vector_space['length-score']) - mean)/std_dev, np.abs(max(vector_space['length-score']) - mean)/std_dev)
    
#Rating mid-sized sentences with higher ratings
vector_space['length-score'] = vector_space['length-score'].apply(lambda val: max_val - np.abs(mean - val)/std_dev)


#-------Summarization Finalized Results--------
#Calculating the final score for each sentence
vector_space['sentence-score'] = vector_space.apply(lambda row: row['tf-score'] + row['length-score'] + row['position-score'], axis = 1)

#Retaining atleast 40% of the original text (in terms of sentences)
num_sentences = int(np.ceil(0.4 * len(sentences)))

scores = sorted(vector_space['sentence-score'].values, reverse = True)

threshold = scores[num_sentences - 1]

summary_sentences = []

for sentence in vector_space.index:
    if vector_space.loc[sentence, 'sentence-score'] >= threshold:
        summary_sentences.append(sentence)
summary = ' '.join(summary_sentences)