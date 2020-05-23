import os
import pandas as pd
import json
from clean_data.clean_data import clean_data
from aspect_extraction.extract_aspects import extract_aspects

def aspect_extraction(nlp, sid, data, text_column, review_id, product_id, folder_path):
    usecols =  ['review_id','review_body','product_id']
    reviews = clean_data(data, text_column = text_column)
    aspect_list = extract_aspects(reviews, nlp, sid, text_column, review_id, product_id)
    """
    aspect_list = list(aspect_list)
    json_path = folder_path + "/reviews_aspect_raw.json"
    with open(json_path, 'w') as outfile:
        json.dump(aspect_list, outfile)
    #reviews = pd.concat([reviews,chunk])

    #print(f'Shape of the merged file:{reviews.shape}')
    """


    return aspect_list
