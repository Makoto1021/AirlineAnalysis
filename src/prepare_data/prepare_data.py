
import pandas as pd
import numpy as np
import datetime
from dateutil import parser


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

def prepare_data():
        
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




    model_path = "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/en_core_web_lg/en_core_web_lg-2.2.5"
    nlp = init_spacy(model_path)
    # -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
    def main(nlp, text_column, review_id, product_id, data, folder_path):
        time1 = time()
        sid = init_nltk()
        time2 = time()
        print("----------------***----------------")
        print("\nExtracting aspect pairs")
        aspect_list = aspect_extraction.aspect_extraction(nlp, sid, data,
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


    folder_path = "/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data"

    aspect_list = main(nlp=nlp, text_column = "text", review_id = 'tweet_id', product_id = 'company', data = df, folder_path = folder_path)

    AS_pairs_df = pd.DataFrame(columns=['product_id', 'review_id', 'noun', 'adj', 'rule', 'polarity_nltk', 'polarity_textblob'])
    for dic in aspect_list:
        new_row = pd.DataFrame({'product_id':[dic["product_id"]], 
                                'review_id':[dic["review_id"]], 
                                'noun':[dic["noun"]], 
                                'adj':[dic["adj"]], 
                                'rule':[dic["rule"]], 
                                'polarity_nltk':[dic["polarity_nltk"]], 
                                'polarity_textblob':[dic["polarity_textblob"]]})
        # new_row = pd.DataFrame.from_dict(dic)
        AS_pairs_df = AS_pairs_df.append(new_row, ignore_index = True)
        
    AS_pairs_df.to_csv("/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data/aspect_sentiment_pairs.csv")

    from nltk.stem import WordNetLemmatizer
    from aspect_clustering.vector_dist import vector_dist
    from run_extraction.init_spacy import init_spacy

    with open("/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data/companies.txt")  as file_in:
        company_list = []
        for line in file_in:
            company_list.append(line.replace('\n', ""))

    company_list_lower = [x.lower() for x in company_list]
    dictionary = dict(zip(company_list_lower, company_list))
    temp = AS_pairs_df[AS_pairs_df.noun.str.lower().isin(company_list_lower)][AS_pairs_df.product_id.str.lower() != AS_pairs_df.noun.str.lower()]
    index_list = temp.index
    df_replace = pd.DataFrame(AS_pairs_df.loc[AS_pairs_df.index.isin(index_list), ["product_id", "noun"]])
    df_replace = df_replace.replace({"noun": dictionary})
    AS_pairs_df.loc[AS_pairs_df.index.isin(index_list), "product_id"] = df_replace.noun

    lemmatizer = WordNetLemmatizer()
    AS_pairs_df['noun_lemmatized'] = AS_pairs_df.noun.str.lower().apply(lemmatizer.lemmatize)
    df_grouped = AS_pairs_df.groupby(["product_id", 'noun_lemmatized']).agg({'product_id':'size',
                                                                    "polarity_nltk":'mean',
                                                                    "polarity_textblob":'mean',
                                                                    "review_id":"unique"}).rename(columns={'product_id':'count',
                                                                                                                'polarity_nltk':'mean_polarity_nltk',
                                                                                                                'polarity_textblob':'mean_polarity_textblob'}).reset_index()

    print("There are %d nouns extracted" %(df_grouped.shape[0]))

    punctuality_vec = nlp('punctuality').vector
    food_vec = nlp('food').vector
    luggage_vec = nlp('luggage').vector
    staff_vec = nlp('staff').vector

    companies = df_grouped.product_id.unique()

    df_grouped['group'] = np.nan

    for company in companies:
        df_sub = df_grouped[df_grouped.product_id == company]
        asp_group = []
        asp_vectors = []
        for aspect in df_sub.noun_lemmatized:
            dist_dic = {}
            token_vector = nlp(aspect).vector
            asp_vectors.append(token_vector)
            # df_grouped[df_grouped.noun_lemmatized==aspect, 'noun_vector'] = token_vector
            dist_dic['punctuality'] = vector_dist(token_vector, punctuality_vec)
            dist_dic['food'] = vector_dist(token_vector, food_vec)
            dist_dic['luggage'] = vector_dist(token_vector, luggage_vec)
            dist_dic['staff'] = vector_dist(token_vector, staff_vec)
            # group = min([dist_punc, dist_food, dist_lugg, dist_staf])
            max_key = max(dist_dic, key=dist_dic.get)
            asp_group.append(max_key)

        df_grouped.loc[df_grouped.product_id == company, "group"] = asp_group
        df_grouped.loc[df_grouped.noun_lemmatized.str.lower() == df_grouped.product_id.str.lower(), "group"] = "company"

        df_vectors_sub = pd.DataFrame(asp_vectors)
        df_vectors_sub['product_id'] = company
        df_vectors_sub['noun_lemmatized'] = df_sub['noun_lemmatized'].values
        df_vectors_sub['count'] = df_sub['count'].values

    df_grouped["tb_importance"] = df_grouped["count"] * df_grouped["mean_polarity_textblob"]

    def weighted_ave(x):
        d = {}
        d['weighted_ave_nltk'] = (x["mean_polarity_nltk"] * x["count"]).sum() / x["count"].sum()
        d['weighted_ave_tb'] = (x["mean_polarity_textblob"] * x["count"]).sum() / x["count"].sum()
        return pd.Series(d, index=['weighted_ave_nltk', 'weighted_ave_tb'])

    df_grouped = df_grouped.groupby(["product_id", "group"]).apply(weighted_ave).reset_index()

    df_grouped.to_csv("/Users/mmiyazaki/Documents/My project/Airline_analysis/src/data/df_grouped.csv")