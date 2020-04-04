from tf_idf_vectorizer import tf_idf_vectorize
import pandas as pd
from bs4 import BeautifulSoup as BS
from nltk import word_tokenize
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from topic_modelling import obtain_key_words
from find_similarity import compute_paragraph_importance
from pre_process_data import pre_process_data

#Reading data from the file
raw_data = ''
clues = []
cue_words = []
with open('source.data', 'r') as source:
    raw_data = source.read()

with open('cue_words.data', 'r') as clues:
    clues = clues.read().split(',')
    
spacy_tool = spacy.load('en_core_web_lg')

for clue in clues:
    for word in clue.split():
        for doc in spacy_tool(word):
            if doc.lemma_ not in spacy_stop_words: cue_words.append(doc.lemma_) 

cue_words = [word.lower() for word in cue_words]
data = BS(raw_data).text

#Preprocessing the data 
sentence_mapper, sentence_paragraph_mapper, finalized_paragraphs = pre_process_data(data, spacy_tool = spacy_tool)

#Obtaining the tf/tf-idf score for the sentences
sentences_with_scores = tf_idf_vectorize(sentence_mapper, spacy_tool=spacy_tool)

original_sentences = list(sentence_mapper.keys())
columns = ['tf-score', 'length-score', 'position-score', 'paragraph-score', 'cue-words-score', 'paragraph_id']

#Extracting paragraph index
# paragraph_index = list(sentence_paragraph_mapper.values())

#Storing all the data in a DataFrame
vector_space = pd.DataFrame(index = original_sentences, columns = columns)

#--------Rating a sentence according to term related metrcis - TF or TFIDF--------
#Noting tf-idf scores of sentences
for sentence, tf_idf_score in sentences_with_scores.items():
    vector_space.loc[sentence, 'tf-score'] = tf_idf_score

#Assigning the paragraph index to the paragraph_id column
# vector_space = vector_space.assign(paragraph_id = paragraph_index)

#Finding the keywords for each paragraph
# keyword_mapper = {}
# for para_id in set(paragraph_index):
#     keywords_with_scores = obtain_key_words(finalized_paragraphs[para_id])
#     keyword_mapper[para_id] = keywords_with_scores
    
# for para_id, keywords in keyword_mapper.items():
#     keyword_mapper[para_id] = compute_paragraph_importance(keywords, cue_words)

#--------Rating a sentence by the importance of the paragraph---------
#Storing the paragraph score in the vector space
# for index, row in vector_space.iterrows():
#     vector_space.loc[index, 'paragraph-score'] = keyword_mapper[row['paragraph_id']]

#--------Rating a sentence by the presence of cuewords---------
for sentence in original_sentences:
    matches = 0
    if len(sentence_mapper[sentence]) == 0:
        vector_space.loc[sentence, 'cue-words-score'] = 0
        continue
    
    for word in cue_words:
        if word in sentence_mapper[sentence]: matches += 1
    vector_space.loc[sentence, 'cue-words-score'] = (matches/len(sentence_mapper[sentence]))
#This metric is very biased - if only 1 cue-word is given then its presence
#will give a score of 1 - highly biases the sentences.
#May be try to put a metric like - length of sentence on the denominator
#and num words which have similarity more than .50 with the cue words - currently only cue-words
    
#----------Rating a sentence according to position and length------------
#Using Barrera and Verma's first model to score sentence based on the position
total_sentences = len(original_sentences)
alpha = 2
    
for index, sentence in enumerate(original_sentences):
    vector_space.loc[sentence, 'position-score'] = (np.cos((2*np.pi*index)/(total_sentences - 1)) + alpha - 1)/(alpha)
    vector_space.loc[sentence, 'length-score'] = len(word_tokenize(sentence))

mean = np.mean(vector_space['length-score'])
std_dev = np.sqrt(np.var(vector_space['length-score']))
max_val = max(np.abs(min(vector_space['length-score']) - mean)/std_dev, np.abs(max(vector_space['length-score']) - mean)/std_dev)
    
#Rating mid-sized sentences with higher ratings
vector_space['length-score'] = vector_space['length-score'].apply(lambda val: max_val - np.abs(mean - val)/std_dev)


#-------Summarization Finalized Results--------
#Calculating the final score for each sentence
#Using - tf-score, length-score, position-score, cue-words-score and the paragraph-score
vector_space['sentence-score'] = vector_space.apply(lambda row: (row['tf-score'] + row['length-score'] + row['position-score'] + row['cue-words-score']), axis = 1)

#Retaining atleast 40% of the original text (in terms of sentences)
num_sentences = int(np.ceil(0.4 * len(original_sentences)))

scores = sorted(vector_space['sentence-score'].values, reverse = True)

threshold = scores[num_sentences - 1]

summary_sentences = []

for sentence in vector_space.index:
    if vector_space.loc[sentence, 'sentence-score'] >= threshold:
        summary_sentences.append(sentence)
summary = ' '.join(summary_sentences)

with open("output.summary" ,'w+') as file:
    file.write(summary)