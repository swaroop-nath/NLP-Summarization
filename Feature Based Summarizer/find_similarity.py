#Necessary imports
import numpy as np
import spacy
from numpy import linalg as LA

'''
    This method computes similarity between the two words using the input tool
'''
def compute_similarity(similarity_tool, word_one, word_two):
    return similarity_tool(word_one).similarity(similarity_tool(word_two))

'''
    This method computes the importance of a paragraph using the similarity
    between keywords from the paragraph and the cue-words input.
'''
def compute_paragraph_importance(spacy_tool = spacy.load('en_core_web_md'), keywords, cue_words):
    importance_matrix = []
    
    for keyword, relevance in keywords.items():
        row = []
        #Exponentiating the convolution of topic importance
        #and its similarity gives better discrimination power.
        for cue_word in cue_words:
            row.append(np.exp(relevance * compute_similarity(spacy_tool, keyword, cue_word)))
            
        importance_matrix.append(row)
    
    return LA.norm(importance_matrix)