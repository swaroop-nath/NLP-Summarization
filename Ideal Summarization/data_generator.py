import os
from bs4 import BeautifulSoup as BS
from aylienapiclient import textapi
import pandas as pd
from nltk.tokenize import sent_tokenize
from tf_idf_vectorizer import tf_idf_vectorize
import numpy as np

data_path = r'../Training Data/' #Appending r in order to tackle the space in the path.

files = []
titles = []

#Collecting path references for all the text files in the Training Data folder.
for _, _, f in os.walk(data_path):
    for file in f:
        files.append(data_path + file)
        titles.append(file.split('.')[0])

app_id = ''
api_key = ''
with open('api_data.config', 'r') as file:
    config_data = file.read().split(',')
    app_id = config_data[0]
    api_key = config_data[1]

#Initializing the Aylien Client to summarize the textual data
client = textapi.Client(app_id, api_key)

#Storing the text data and summaries for all the text files in respective dicts for later use.
summaries = {}
text_data = {}

index = 0

for file_name in files:
    with open(file_name) as file:
        raw_data = file.read()
        data = BS(raw_data, features='lxml').text
        text_data[file_name] = data
        summary = client.Summarize({'text': data, 'title': titles[index]})
        summary = summary['sentences']
        summaries[file_name] = summary
        index += 1

#Vectorizing the text data of each file
features = ['TF score', 'Relative Location', 'Relative Length', 'Label']

sentences = []

for data in text_data.values():
    for sentence in sent_tokenize(data):
        sentences.append(sentence)
    
temp_data = [[0]*4]*len(sentences)

df = pd.DataFrame(data = temp_data, index = sentences, columns = features)

for file_name, data in text_data.items():
    sentences = sent_tokenize(data)
    scores = tf_idf_vectorize(data)
    
    total_sentences = len(sentences)
    total_characters = len(data)
    
    for index, sentence in enumerate(sentences):
        df.loc[sentence, features[0]] = scores[index]
        df.loc[sentence, features[1]] = (index + 1)/total_sentences
        df.loc[sentence, features[2]] = len(sentence)/total_characters
        
        if sentence in summaries[file_name]:
            df.loc[sentence, features[3]] = 1
        else:
            df.loc[sentence, features[3]] = 0
        
df.to_csv('../training_data.csv', sep = ',')

#For verification purpose only
for index, summary in enumerate(summaries.values()):
    with open('../' + titles[index] + '.txt', 'w') as file:
        file.write(' '.join(summary))