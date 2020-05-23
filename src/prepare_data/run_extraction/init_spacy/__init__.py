import spacy
import ast


l = []

with open("/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data/stopwords.txt")  as file_in:
    stopwords = []
    for line in file_in:
        stopwords.append(line.replace('\n', ""))

with open("/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data/exclude_stopwords.txt")  as file_in:
    exclude_stopwords = []
    for line in file_in:
        exclude_stopwords.append(line.replace('\n', ""))

def init_spacy(model_path):
    print("\nLoading spaCy Model....")
    nlp = spacy.load(model_path)
    print("spaCy successfully loaded")
    for w in stopwords: # for the words in stopwords, set "is_stop == True" in spacy model
        nlp.vocab[w].is_stop = True
    for w in exclude_stopwords: # for the words in exclude_stopwords, set "is_stop == False" in spacy model
        nlp.vocab[w].is_stop = False
    return nlp