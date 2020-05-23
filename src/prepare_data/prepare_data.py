
import pandas as pd
import numpy as np
import datetime
from dateutil import parser

# Parse timestamp column
df = pd.read_csv('/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data/raw_tweets.csv')
df["timestamp"] = df["timestamp"].apply(parser.parse)
print("timestamp parsed")

# Remove URL
import re
import unicodedata

df['text'] = df.text.str.replace(u'\xa0', u' ')

def remove_URL(text):
    text = re.sub(r"""((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))*))+(?:(([^\s()<>]+|(([^\s()<>]+)))*)|[^\s`!()[]{};:'".,<>?«»“”‘’]))""", "", text)
    text = re.sub(r'twitter\.com\S+', '', text)
    return text

df['text']=df.text.apply(remove_URL)
print("URL removed")

# Tidy up links
df["links"]=df["links"].str.replace("[", "").replace("]", "").replace("'", "")
print("links tidied up")



# Normalize text
df["text"] = df["text"].str.lower()
print("text normalized")

# remove short tweets
df["length"] = df["text"].apply(len)
df = df[df["length"]>=55]
print("short tweets removed")


import os
import sys
import spacy
from time import time
import json
import requests
from aspect_extraction.aspect_extraction import aspect_extraction
import mapper
from run_extraction.init_spacy import init_spacy
from run_extraction.init_nltk import init_nltk



# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def main(text_column, review_id, product_id, data, folder_path):
    model_path= dataiku.get_custom_variables()['model_path']
    time1 = time()
    nlp = init_spacy(model_path)
    sid = init_nltk()
    time2 = time()
    print("----------------***----------------")
    print("\nExtracting aspect pairs")
    aspect_list = aspect_extraction(nlp, sid, data,
                                        text_column = text_column,
                                        review_id = review_id,
                                        product_id = product_id,
                                       folder_path = folder_path)
    # print(aspect_list)
    print("Finished running aspect extraction!!\n")

    # json_data = json.dumps(reviews_data)
    # with open('data.json', 'w') as outfile:
    #     f.write(json_data)

    # print("----------------***----------------")
    # time3 = time()
    # aspect_clustering.update_reviews_data(reviews_data, nlp)
    time4 = time()
    print("Time for spacy loading: {0:.2}s".format(time2-time1))
    # print("Time for aspect extraction: {0:.2}s".format(time3-time2))
    print("Time for EVERYTHING: {0:.2}s".format(time4-time1))
    print("Running mapper")
    
    """

    file_in = folder_path + "/reviews_aspect_raw.json"
    file_out = folder_path + "/reviews_aspect_mapping.json"
    mapper.map(file_in, file_out)
    """

    print("Godspeed!")
    return aspect_list


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
# aspect_sentiment_pairs = dataiku.Folder("tweepy_aspect_sentiment_pairs")
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
folder_path = "/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
aspect_list = main(text_column = "text", review_id = 'tweet_id',
     product_id = 'company', data = df, folder_path = folder_path)

"""
processed = []
with open(folder_path + "/reviews_aspect_mapping.json") as f:
    for l in f:
        processed.append(json.loads(l.strip()))
        
tweet_processed = pd.DataFrame(columns=['product_id', 'review_id', 'noun', 'adj', 'rule', 'polarity_nltk', 'polarity_textblob'])

for p in processed[0]:
    prod_id = [i for i in iter(p.keys())][0]
    print(prod_id)
    reviews = p[prod_id]
    # print(reviews)
    for review in reviews:
        review_id = review['review_id']
        for asp in review['aspect_pairs']:
            noun = asp['noun']
            adj = asp['adj']
            rule = asp['rule']
            polarity_nltk = asp['polarity_nltk']
            polarity_textblob = asp['polarity_textblob']
            new_row = pd.DataFrame({'product_id':[prod_id], 'review_id':[review_id], 
                                    'noun':[noun], 'adj':[adj], 'rule':[rule], 
                                    'polarity_nltk':[polarity_nltk], 
                                    'polarity_textblob':[polarity_textblob]})
            tweet_processed = tweet_processed.append(new_row, ignore_index=True)



py_recipe_output = dataiku.Dataset("aspect_sentiment_pairs")
py_recipe_output.write_with_schema(tweet_processed)
"""
tweet_processed = pd.DataFrame(columns=['product_id', 'review_id', 'noun', 'adj', 'rule', 'polarity_nltk', 'polarity_textblob'])
for dic in aspect_list:
    new_row = pd.DataFrame({'product_id':[dic["product_id"]], 
                            'review_id':[dic["review_id"]], 
                            'noun':[dic["noun"]], 
                            'adj':[dic["adj"]], 
                            'rule':[dic["rule"]], 
                            'polarity_nltk':[dic["polarity_nltk"]], 
                            'polarity_textblob':[dic["polarity_textblob"]]})
    # new_row = pd.DataFrame.from_dict(dic)
    tweet_processed = tweet_processed.append(new_row, ignore_index = True)
    
py_recipe_output = dataiku.Dataset("tweepy_aspect_sentiment_pairs")
py_recipe_output.write_with_schema(tweet_processed)