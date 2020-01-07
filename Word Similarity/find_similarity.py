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
def compute_paragraph_importance(keywords, cue_words):
    importance_matrix = []
    spacy_tool = spacy.load('en_core_web_lg')
    
    for keyword, relevance in keywords.items():
        row = []
        for cue_word in cue_words:
            row.append(np.exp(relevance * compute_similarity(spacy_tool, keyword, cue_word)))
            
        importance_matrix.append(row)
    
    return LA.norm(importance_matrix)