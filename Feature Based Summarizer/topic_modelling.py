#Imports
import re
import os
import string
import matplotlib.pyplot as plt
import spacy
import gensim.corpora as corpora
from gensim.models import ldamodel, CoherenceModel
from gensim.models.wrappers import LdaMallet
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from wordcloud import WordCloud
from sklearn.model_selection import GridSearchCV

#Defining constants
TOPICS_LIMIT = 10
NUM_PASSES = 100
FRACTION_OUTPUT = 0.25

#Pipelines
'''
    This method generates a list of stop words.
'''
def get_stop_words(documents):
    stop_word_list = [word for word in spacy_stop_words]
    return stop_word_list
    
  
'''
    This method does removal of unnecessary notations (punctuations, apostrophe etc.), and tokenization.
'''
def generate_tokenized_documents(documents):
    non_space_regex = '[0-9\[\]%/,()–“”\']'
    tokenized_documents = []
    
    for doc in documents:
        cleaned_doc = re.sub(non_space_regex, '', doc).translate(str.maketrans('', '', string.punctuation))
        word_tokens = word_tokenize(cleaned_doc)
        word_tokens = [word.lower() for word in word_tokens]
        tokenized_documents.append(word_tokens)
        
    return tokenized_documents
    
'''
    This method lemmatizes the documents.
'''
def generate_lemmatized_documents(tool, documents, allowed_pos_tags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
    lemmatized_documents = []
    for doc in documents:
        lemma_generator = tool(' '.join(doc))
        lemmatized_sentence = [token.lemma_ for token in lemma_generator if len(token) > 2]
        lemmatized_documents.append(lemmatized_sentence)
    return lemmatized_documents
   
'''
This method generates a cleaned document free of stop words.
'''
def remove_stop_words(stop_word_list, documents):
    cleaned_documents = []
    for doc in documents:
        cleaned_doc = [word for word in doc if word not in stop_word_list]
        cleaned_documents.append(cleaned_doc)
    return cleaned_documents

def find_optimum_topics(corpus, final_documents, word_dict):
    topics_wise_score = {}
    for num_topics in range(1, TOPICS_LIMIT):
        lda_model = ldamodel.LdaModel(corpus = corpus, random_state = 100, 
                                            id2word = word_dict, passes = NUM_PASSES, num_topics = num_topics)
        coherence_score = CoherenceModel(model = lda_model, texts = final_documents, dictionary = word_dict, coherence = 'c_v').get_coherence()
        topics_wise_score[num_topics] = coherence_score
        
    leader = -1
    leader_score = -1
    for num_topics, score in topics_wise_score.items():
        if score > leader_score: 
            leader = num_topics
            leader_score = score
    
    return leader, leader_score

def build_lda_model(corpus, final_documents, word_dict):
    num_topics, optimum_score = find_optimum_topics(corpus, final_documents, word_dict)
    
    alpha_pool = [val * 0.05 for val in list(range(1, 21))]
    eta_pool = [val * 0.05 for val in list(range(1, 21))]
    
    alpha_wise_score = {}
    eta_wise_score = {}
    
    for alpha in alpha_pool:
        lda_model = ldamodel.LdaModel(corpus = corpus, random_state = 100, 
                                            id2word = word_dict, passes = NUM_PASSES, num_topics = num_topics, alpha = alpha)
        coherence_score = CoherenceModel(model = lda_model, texts = final_documents, dictionary = word_dict, coherence = 'c_v').get_coherence()
        alpha_wise_score[alpha] = coherence_score
    
    leading_alpha = -1
    leading_alpha_score = -1
    for alpha, score in alpha_wise_score.items():
        if score > leading_alpha_score and score > optimum_score:
            leading_alpha = alpha
            leading_alpha_score = score
            
    if leading_alpha == -1: leading_alpha = 'auto'
    if leading_alpha_score != -1: optimum_score = leading_alpha_score
    
    for eta in eta_pool:
        lda_model = ldamodel.LdaModel(corpus = corpus, random_state = 100, 
                                            id2word = word_dict, passes = NUM_PASSES, num_topics = num_topics, alpha = leading_alpha, eta = eta)
        coherence_score = CoherenceModel(model = lda_model, texts = final_documents, dictionary = word_dict, coherence = 'c_v').get_coherence()
        eta_wise_score[eta] = coherence_score
    
    leading_eta = -1
    leading_eta_score = -1
    for eta, score in eta_wise_score.items():
        if score > leading_eta_score and score > optimum_score:
            leading_eta = eta
            leading_eta_score = score
            
    if leading_eta == -1: leading_eta = 'auto'
    if leading_eta_score != -1: optimum_score = leading_eta_score
    
    best_lda_model = ldamodel.LdaModel(corpus = corpus, random_state = 100, 
                                            id2word = word_dict, passes = NUM_PASSES, num_topics = num_topics, alpha = leading_alpha, eta = leading_eta)
    return best_lda_model
        

def obtain_key_words(data):
    #Intializing spacy tool
    spacy_tool = spacy.load('en', disable=['parser', 'ner'])
        
    #Tokenize into sentences
    documents = sent_tokenize(data)
    
    #Removing punctuations and generating tokenized documents
    tokenized_documents = generate_tokenized_documents(documents)
    
    #Getting stop words based on the context and removing them
    stop_word_list = get_stop_words(tokenized_documents)
    cleaned_documents = remove_stop_words(stop_word_list, tokenized_documents)
    
    #Generating lemmatized documents
    final_documents = generate_lemmatized_documents(spacy_tool, cleaned_documents)
    
    #Creating a dictionary of words
    word_dict = corpora.Dictionary(final_documents)
    
    #Creating term-document frequency for Gensim LDA
    corpus = [word_dict.doc2bow(document) for document in final_documents]
    
    best_model = build_lda_model(corpus, final_documents, word_dict)
    
    topics = [dict(topic[1]) for topic in best_model.show_topics(formatted = False)]
    
    keywords = {}
    
    for topic in topics:
        for word, importance in topic.items():
            if keywords.get(word) == None: keywords[word] = importance
            else: keywords[word] += importance
    
    num_words_output = int(FRACTION_OUTPUT * len(tokenized_documents))
    
    keywords = sorted(keywords.items(), key = lambda item: item[1], reverse = True)
    
    #Returning top 25% keywords in the paragraph with value above 1 for each word
    output = keywords[0: num_words_output]
    for index in range(len(output)):
        output[index] = list(output[index])
        output[index][1] = output[index][1] + 1
        output[index] = tuple(output[index])
        
    return output
    
