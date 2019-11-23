from aylienapiclient import textapi
from bs4 import BeautifulSoup as BS

app_id = ''
api_key = ''
with open('api_data.config', 'r') as file:
    config_data = file.read().split(',')
    app_id = config_data[0]
    api_key = config_data[1]

#Initializing the Aylien Client to summarize the textual data
client = textapi.Client(app_id, api_key)

# Importing the dataset
raw_data = ''
with open('../data.txt', 'rb') as file:
    raw_data = file.read()

# Cleaning the data
data = BS(raw_data, features='lxml').text

summary = client.Summarize({'text': data, 'title': 'Isaac Newton'})

summary = ' '.join(summary['sentences'])

with open('output.txt', 'w') as file:
    file.write(summary)