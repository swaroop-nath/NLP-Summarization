from tf_idf_vectorizer import tf_idf_vectorize
import pandas as pd
from bs4 import BeautifulSoup as BS
from nltk import sent_tokenize, word_tokenize
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from topic_modelling import obtain_key_words

#Pipeline
def map_sentence_to_paragraph(vector_space, sentences, paragraphs):
    for sentence in sentences:
        for index, paragraph in enumerate(paragraphs):
            if sentence in paragraph: vector_space.loc[sentence, 'paragraph_id'] = index

#Reading data from the file
raw_data = ''
clues = []
cue_words = []
with open('source.data', 'r') as source:
    raw_data = source.read()

with open('cue_words.data', 'r') as clues:
    clues = clues.read().split(',')
    
lemmatizer = spacy.load('en_core_web_lg')

for clue in clues:
    for word in clue.split():
        for doc in lemmatizer(word):
            if doc.lemma_ not in spacy_stop_words: cue_words.append(doc.lemma_) 
    
data = BS(raw_data).text

#Obtaining the tf/tf-idf score for the sentences
sentences_with_scores = tf_idf_vectorize(data)

columns = ['tf-score', 'length-score', 'position-score', 'paragraph-score', 'cue-words-score', 'paragraph_id']

paragraphs = data.split('\n')
sentences = sent_tokenize(data)

#Storing all the data in a DataFrame
vector_space = pd.DataFrame(index = sentences, columns = columns)

#Noting tf-idf scores of sentences
for sentence, tf_idf_score in sentences_with_scores.items():
    vector_space.loc[sentence, 'tf-score'] = tf_idf_score
    
#Storing paragraphs and noting there scores
map_sentence_to_paragraph(vector_space, sentences, paragraphs)

for paragraph in paragraphs:
    keywords = obtain_key_words(paragraph)
    print(keywords)
    
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