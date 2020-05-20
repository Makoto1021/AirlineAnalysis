import pandas as pd
from sys import argv
import re
from clean_data.demojify import demojify
import dataiku

#Input file name as argv to run

#Achyut - Commenting this out because repeated
# def read_data():
#     return pd.read_csv(file_name,
#         sep = "\t", header=0, error_bad_lines = False)



def clean_data(df, text_column):
    # text_column: column name of the text

    pd.options.mode.chained_assignment = None

    print("******Cleaning Started*****")

    print(f'Shape of df before cleaning : {df.shape}')
    #df['review_date'] = pd.to_datetime(df['review_date'])
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].str.replace("<br />", " ")
    df[text_column] = df[text_column].str.replace("\[?\[.+?\]?\]", " ")
    df[text_column] = df[text_column].str.replace("\/{3,}", " ")
    df[text_column] = df[text_column].str.replace("\&\#.+\&\#\d+?;", " ")
    df[text_column] = df[text_column].str.replace("\d+\&\#\d+?;", " ")
    df[text_column] = df[text_column].str.replace("\&\#\d+?;", " ")
    df[text_column] = df[text_column].str.replace("\\", " ")
    df[text_column] = df[text_column].str.replace("@", " ")
    df[text_column] = df[text_column].str.replace("#", " ")

    #facial expressions
    df[text_column] = df[text_column].str.replace("\:\|", "")
    df[text_column] = df[text_column].str.replace("\:\)", "")
    df[text_column] = df[text_column].str.replace("\:\(", "")
    df[text_column] = df[text_column].str.replace("\:\/", "")
    df[text_column] = df[text_column].str.replace("\;\|", "")
    df[text_column] = df[text_column].str.replace("\;\)", "")
    df[text_column] = df[text_column].str.replace("\;\(", "")
    df[text_column] = df[text_column].str.replace("\;\/", "")
    
    # removing emojis
    df[text_column]=df[text_column].apply(demojify)
    

    #replace multiple spaces with single space
    df[text_column] = df[text_column].str.replace("\s{2,}", " ")

    df[text_column] = df[text_column].str.lower()
    print(f'Shape of df after cleaning : {df.shape}')
    print("******Cleaning Ended*****")


    return(df)
#
# if __name__ == '__main__':
#     #file_name = str(argv[1])
#     #df = read_data()
#
#
#     out_name = "CLEANED_"+file_name
#
#     df.to_csv(out_name, sep='\t', index=False)
