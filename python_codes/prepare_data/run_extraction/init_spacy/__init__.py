import spacy
import dataiku
import ast

stopwords = dataiku.get_custom_variables(typed=True)['stopwords']

exclude_stopwords = dataiku.get_custom_variables(typed=True)['exclude_stopwords']

def init_spacy(model_path):
    print("\nLoading spaCy Model....")
    nlp = spacy.load(model_path)
    print("spaCy successfully loaded")
    for w in stopwords: # for the words in stopwords, set "is_stop == True" in spacy model
        nlp.vocab[w].is_stop = True
    for w in exclude_stopwords: # for the words in exclude_stopwords, set "is_stop == False" in spacy model
        nlp.vocab[w].is_stop = False
    return nlp