# Basic imports 
import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import numpy as np
import string
from pprint import pprint

# Importing gensim utilities
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
from gensim.models.wrappers import LdaMallet

# Importing spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Lemmatizing the document sent to the method
def lemmatize_document(lemmatizer, document, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
    lemma_generator = lemmatizer(document)
    
    lemmatized_words = [token.lemma_ for token in lemma_generator if token.pos_ in allowed_postags]
    
    return lemmatized_words

#Pre-process data
def pre_process_document(document):
    #Removing non-alphabetical characters
    non_space_regex = '[0-9\[\]%/,()–“”\']'
    clean_document = re.sub(non_space_regex, '', document)
    
    #Removing punctuations
    clean_document = clean_document.translate(str.maketrans('', '', string.punctuation))
    
    #Tokenizing the document
    word_tokens = word_tokenize(clean_document)
    
    #Lower-casing all the words
    word_tokens = [word.lower() for word in word_tokens]
    
    #Removing stop-words
    stop_words = stopwords.words('english')
    final_words = [word for word in word_tokens if word not in stop_words]
    
#    #Lemmatizing the words
#    final_document = lemmatize_document(lemmatizer, ' '.join(final_words), allowed_postags = ['NOUN', 'ADJ'])
#    
##    Making bi-grams from lemmatized words
##    for first, second in zip(final_document[:-1], final_document[1:]):
##        final_document.append('_'.join([first, second]))
#    
#    #Removing small words (less than two in size)
#    final_n_grams = [word for word in final_document if len(word) > 2]
#    return final_n_grams
    
    return final_words

def make_bigrams(phraser, documents):
    return [phraser[document] for document in documents]

#Reading the textual data
data = ''
with open('document.data', encoding='utf8') as file:
    data = file.read()

documents = sent_tokenize(data)

spacy_lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

cleaned_documents = []

for document in documents:
    cleaned_documents.append(pre_process_document(document))
    
#Creating bigrams in the document
#phrases = Phrases(cleaned_documents, min_count=1, threshold=1)
#phraser = Phraser(phrases)
#cleaned_documents = make_bigrams(phraser, cleaned_documents);

#Lemmatizing the document
final_documents = []
for document in cleaned_documents:
    final_documents.append([word for word in lemmatize_document(spacy_lemmatizer, ' '.join(document), allowed_postags=['NOUN', 'ADJ']) if len(word) > 2])
    
#Creating a dictionary of words
word_dict = corpora.Dictionary(final_documents)

#Creating term-document frequency for Gensim LDA
corpus = [word_dict.doc2bow(document) for document in final_documents]

#Building the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus, random_state = 100, 
                                            id2word = word_dict, passes = 50, num_topics = 1)

pprint(lda_model.print_topics())

#Evaluating performance metric - Coherence Score
coherence_model_lda = CoherenceModel(model = lda_model, texts = final_documents, dictionary = word_dict, coherence = 'c_v')
coherence_score = coherence_model_lda.get_coherence()
print(coherence_score)

#Using latent Semantic Analysis
lsi_model = gensim.models.lsimodel.LsiModel(corpus = corpus, id2word = word_dict, num_topics = 1)

pprint(lsi_model.print_topics())

#Evaluating performance metric for LSI Model
coherence_model_lda = CoherenceModel(model = lsi_model, texts = final_documents, dictionary = word_dict, coherence = 'c_v')
coherence_score = coherence_model_lda.get_coherence()
print(coherence_score)

#Displaying word-cloud
topics = lda_model.show_topics(formatted = False)

for topic in topics:
    topic_words = dict(topic[1])
    cloud = WordCloud(background_color = 'white')
    cloud.generate_from_frequencies(topic_words, max_font_size = 300)
    plt.figure()
    plt.imshow(cloud)

#Getting document-wise topic distribution
doc_topic_data = pd.DataFrame()

for index, document in enumerate(corpus):
    topic_dist = lda_model[document]
    topic_0_perc = topic_dist[0][1]
    topic_1_perc = topic_dist[1][1]
    
    doc_topic_data = doc_topic_data.append(pd.Series([topic_0_perc, topic_1_perc, documents[index]]), ignore_index = True)
    
doc_topic_data.columns = ['Topic_0_Perc', 'Topic_1_Perc', 'Document']

# Topic Modelling using Mallet
os.environ['MALLET_HOME'] = r'C:\Users\swanath\mallet-2.0.8'
mallet_path = r'C:\Users\swanath\mallet-2.0.8\bin\mallet.bat'
lda_mallet = LdaMallet(mallet_path, corpus=corpus, num_topics=1, id2word=word_dict)

pprint(lda_mallet.show_topics(formatted=False))

#Evaluating performance metric - Coherence Score
coherence_model_lda = CoherenceModel(model = lda_mallet, texts = cleaned_documents, dictionary = word_dict, coherence = 'c_v')
coherence_score = coherence_model_lda.get_coherence()
print(coherence_score)