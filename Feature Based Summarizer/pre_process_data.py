#Necessary imports
import re, string
from nltk.tokenize import word_tokenize

#preprocessing Pipelines
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

'''
    Preprocesses the entire textual document and returns two dicts-
    dict_1: key - sentence, value - paragraph_id
    dict_2: key - paragraph_id, value - paragraph
    paragraph is a 1D array of sentences, which in turn is a 1D array of words.
'''
def pre_process_data(text_corpus, spacy_tool = spacy.load('en_core_web_lg'), stop_words):
    pass