import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from numpy import log

'''
    TF-IDF Vectorizer method.
    For use, pass the text data, and the kind of vectorization
    desired (tf or tf-idf)
'''
def tf_idf_vectorize(sentence_mapper, mode = 'tf', spacy_tool = spacy.load('en_core_web_lg'), stop_words = spacy_stop_words):
    if not (mode == 'tf' or mode == 'tf-idf'):
        raise InvalidVectorizationTechniqueError('Invalid mode chosen.')

    original_sentences = list(sentence_mapper.keys())
    processed_sentences = list(sentence_mapper.values())
    
    words = [word for sentence in processed_sentences for word in sentence]

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

    sentence_scores = {}

    #Checking for the mode of vectorization
    if mode == 'tf':
        for sent in original_sentences:
            sentence_scores[sent] = 0
            
        for i, sentence in enumerate(processed_sentences):
            for word in sentence:
                sentence_scores[original_sentences[i]] += term_freq_matrix[word]
    else:
        # Calculating Inverse Document Frequency - treating each sentence as a document
        num_docs = len(original_sentences)

        idf_matrix = {}

        for word in words:
            if idf_matrix.get(word) == None:
                count = 0
                for sent in processed_sentences:
                    if word in sent: count += 1
                idf_matrix[word] = count

        for key in idf_matrix.keys():
            idf_matrix[key] = log(num_docs/idf_matrix[key])
        
        # Calculating the TFIDF score for each word
        tf_idf_score = {}

        for key in term_freq_matrix.keys():
            tf_idf_score[key] = term_freq_matrix[key] * idf_matrix[key]

        # Scoring sentences
        for sent in original_sentences:
            sentence_scores[sent] = 0
            
        for i, sentence in enumerate(processed_sentences):
            for word in sentence:
                sentence_scores[original_sentences[i]] += tf_idf_score[word]

    #Returning the score of each sentence
    return sentence_scores

class InvalidVectorizationTechniqueError(Exception):
    message = ''
    def __init__(self, message):
        self.message = message

    def getMessage(self):
        return self.message