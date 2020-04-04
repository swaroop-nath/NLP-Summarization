#Necessary imports
import re, string
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words

#preprocessing Pipelines
'''
    This method does removal of unnecessary notations (punctuations, apostrophe etc.), and tokenization.
'''
def generate_tokenized_documents(paragraph):
    non_space_regex = '[0-9\[\]%/,()–“”\']'
    tokenized_documents = []
    
    for doc in paragraph:
        cleaned_doc = re.sub(non_space_regex, '', doc).translate(str.maketrans('', '', string.punctuation))
        word_tokens = word_tokenize(cleaned_doc)
        word_tokens = [word.lower() for word in word_tokens]
        tokenized_documents.append(word_tokens)
        
    return tokenized_documents
    
'''
    This method lemmatizes the documents.
'''
def generate_lemmatized_documents(tool, paragraph, allowed_pos_tags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
    lemmatized_documents = []
    for doc in paragraph:
        lemma_generator = tool(' '.join(doc))
        lemmatized_sentence = [token.lemma_ for token in lemma_generator if len(token) > 2]
        lemmatized_documents.append(lemmatized_sentence)
    return lemmatized_documents
   
'''
This method generates a cleaned document free of stop words.
'''
def remove_stop_words(stop_word_list, paragraph):
    cleaned_documents = []
    for doc in paragraph:
        cleaned_doc = [word for word in doc if word not in stop_word_list]
        cleaned_documents.append(cleaned_doc)
    return cleaned_documents

'''
    Preprocesses the entire textual document and returns three dicts-
    dict_1: original_sentence, value - finalized_sentence
    dict_2: key - original_sentence, value - paragraph_id
    dict_3: key - paragraph_id, value - finalized_paragraph
    paragraph is a 1D array of sentences, which in turn is a 1D array of words.
'''
def pre_process_data(text_corpus, spacy_tool = spacy.load('en_core_web_lg'), stop_word_list = spacy_stop_words):
    # First step - extract original sentences and map each to a paragraph
    paragraphs = text_corpus.split('\n')
    original_sentences = []
    
    dict_1 = {}
    dict_2 = {}
    finalized_paragraphs = []
    
    for index, paragraph in enumerate(paragraphs):
        original_sentences.append(sent_tokenize(paragraph))
        
        documents = sent_tokenize(paragraph)
        
        tokenized_paragraph = generate_tokenized_documents(documents)
        cleaned_paragraph = remove_stop_words(stop_word_list, tokenized_paragraph)
        finalized_paragraph = generate_lemmatized_documents(spacy_tool, cleaned_paragraph)
        
        for sentence_index, original_sentence in enumerate(original_sentences[index]):
            dict_1[original_sentence] = finalized_paragraph[sentence_index]
            dict_2[original_sentence] = index
        
        finalized_paragraphs.append(finalized_paragraph)
            
    
    return dict_1, dict_2, finalized_paragraphs