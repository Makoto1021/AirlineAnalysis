import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import time
import random
import ast

score_table = "tweepy_analysis_by_companies"
tweet_id_table = "tweepy_aspect_sentiment_categorised"
tweets_table = "URL_removed"



# function to get the dataframe based on the choice from the dropdown
# columns included: 'group', 'weighted_ave_tb'
def get_dataset_selection(company):
    df = pd.read_csv(score_table)
    # df = df[df.product_id == company][['group', 'weighted_ave_tb']]
    new = df["product_id"].isin(company)
    df = df[new][['product_id', 'group', 'weighted_ave_tb']]
    print("df.columns", df.columns)
    print("df.product_id.unique()", df.product_id.unique())
    return df

@app.route('/get_filter_values')
# function to get unique values for airline companies to show in drop down
def get_filter_values():
    df = pd.read_csv(score_table)
    companies_list = df['product_id'].unique().tolist()
    print(companies_list)
    return json.dumps({'companies': companies_list})

@app.route('/get_stats/<path:params>')
def get_stats(params):
    params_dict = json.loads(params)
    selected_companies = params_dict.get('companies')
    print("selected_companies is", selected_companies)
    df = get_dataset_selection(selected_companies)
    # bar_chart = df.to_dict(orient='records')
    # bar_chart = {"data":df["weighted_ave_tb"].tolist(), "labels":df["group"].tolist()}
    # return json.dumps({'chart':bar_chart})
    global_label = df["group"].unique().tolist()
    l = []
    for company in df.product_id.unique():
        sentiment_score_list = []
        sub_df = df[df.product_id == company]
        
        for g in global_label: # for example, Emirates doesn't have "company" score
            if g in sub_df.group.unique():
                value = sub_df[sub_df.group==g].weighted_ave_tb.values[0]
            else:
                value = 0
            sentiment_score_list.append(value)
        
        d = {company:{
            # "labels":sub_df["group"].tolist(), 
                          "data":sentiment_score_list}}
        l.append(d)
    bar_chart_group = {"company":l, "labels":global_label}
    
    return json.dumps({'barChartGroup':bar_chart_group})
    
@app.route('/get_table/<path:params>')
def get_table(params):
    params_dict = json.loads(params)
    selected_company = params_dict.get('company')[0]
    selected_category = params_dict.get('category')
    df = pd.read_csv(tweet_id_table)
    df_sorted = df[(df.product_id == selected_company) & (df.group == selected_category)].sort_values("tb_importance", ascending = False)
    df_sorted = df_sorted[["noun_lemmatized", "mean_polarity_textblob", "tb_importance", "review_id"]].rename(columns={"noun_lemmatized":"topics", "mean_polarity_textblob":"average scores", "tb_importance":"importance scores"})
    top5_pos = df_sorted.head(5).reset_index(drop=True).to_json(orient='index')
    top5_neg = df_sorted.tail(5).reset_index(drop=True).to_json(orient='index')
    # top5_pos = top5_pos.to_html(classes=['table', 'table-bordered'], index=False, na_rep='')
    # top5_neg = top5_neg.to_html(classes=['table', 'table-bordered'], index=False, na_rep='')
    data = { 'table_pos' : top5_pos, 'table_neg' : top5_neg }
    json.dumps(data)
    # return json.dumps({'table_pos':top5_pos, "table_neg":top5_neg})
    return json.dumps(data)
    
    
    
    
@app.route('/get_tweets_table/<path:params>')
def get_tweets_table(params):
    params_dict = json.loads(params)
    company = params_dict.get('company')[0]
    topic = params_dict.get('topic')
    print("topic is", topic)
    tweet_id = params_dict.get('review_id').replace(" ", ", ").replace("/n", "")
    print("tweet_id:", tweet_id)
    tweet_id = eval(tweet_id)
    print("type of ids", type(tweet_id))
    if len(tweet_id) > 5:
        id_random_select = random.sample(tweet_id, 5)
    else:
        id_random_select = tweet_id
    
    id_random_select = map(str, id_random_select)
    print("randomly selected tweet ids ", id_random_select)
    df = pd.read_csv(tweets_table)
    df = df[['timestamp', "company", 'tweet_id', 'text', 'username', 'user_location', 'user_id']]
    df['tweet_id'] = df.tweet_id.astype(str)
    df['timestamp'] = df.timestamp.astype(str)
    print("columns", df.shape)
    df = df[(df["company"]==company) & (df["tweet_id"].isin(id_random_select))].drop("company", axis=1).reset_index(drop=True).to_json(orient='index')
    print("this is how tweets are returned to JS", df)
    return df
    
    
    
@app.route('/compare_companies_from_nouns/<path:params>')
def compare_companies_from_nouns(params):
    params_dict = json.loads(params)
    topic = params_dict.get('topic')
    df = pd.read_csv(tweet_id_table)
    df_filtered = df[df.noun_lemmatized == topic][['product_id', 'noun_lemmatized', 'mean_polarity_textblob']].reset_index(drop=True)
    # df_filtered = df_filtered.to_json(orient='index')
    df_dict = df_filtered.to_dict('records')
    print("df_dict", df_dict)
    df_filtered_json = json.dumps({'barNounChart':df_dict})
    print("df_filtered_json", df_filtered_json)
    return df_filtered_json