import os
import pandas as pd
import json
from clean_data.clean_data import clean_data
from aspect_extraction.extract_aspects import extract_aspects

def aspect_extraction(nlp, sid, data, text_column, review_id, product_id, folder_path):
    usecols =  ['review_id','review_body','product_id']
    print("entered clean_data")
    reviews = clean_data(data, text_column = text_column)
    aspect_list = extract_aspects(reviews, nlp, sid, text_column, review_id, product_id)
    print("it's ok until here")

    return aspect_list
