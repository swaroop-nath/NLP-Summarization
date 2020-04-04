#Imports
import spacy
import gensim.corpora as corpora
from gensim.models import ldamodel, CoherenceModel

#Defining constants
TOPICS_LIMIT = 10
NUM_PASSES = 100
FRACTION_OUTPUT = 0.25

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
        

def obtain_key_words(paragraph, spacy_tool = spacy.load('en_core_web_lg', disable = ['parser', 'ner'])):
    #Creating a dictionary of words
    word_dict = corpora.Dictionary(paragraph)
    
    #Creating term-document frequency for Gensim LDA
    corpus = [word_dict.doc2bow(document) for document in paragraph]
    
    best_model = build_lda_model(corpus, paragraph, word_dict)
    
    topics = [dict(topic[1]) for topic in best_model.show_topics(formatted = False)]
    
    keywords = {}
    
    for topic in topics:
        for word, importance in topic.items():
            if keywords.get(word) == None: keywords[word] = importance
            else: keywords[word] += importance
    
    num_words_output = int(FRACTION_OUTPUT * len(keywords))
    
    keywords = sorted(keywords.items(), key = lambda item: item[1], reverse = True)
    
    #Returning top 25% keywords in the paragraph with value above 1 for each word
    output = keywords[0: num_words_output]
    
    #Exponentiating the importance - doesn't really add to the compound discriminating
    #power
#    for index in range(len(output)):
#        output[index] = list(output[index])
#        output[index][1] = math.exp(output[index][1])
#        output[index] = tuple(output[index])
        
    return output