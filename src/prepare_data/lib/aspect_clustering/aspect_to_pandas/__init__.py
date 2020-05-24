import pandas as pd

def aspect_to_pandas(reviews):
    tweet_clustered = pd.DataFrame(columns=['product_id', 'review_id', 'noun', 'adj', 'rule', 'polarity_nltk', 'polarity_textblob', 'cluster'])
    # print("prod_id in aspect_to_pandas is", prod_id)
    # print(reviews[0])
    for i, product in enumerate(reviews):
        # print(i)
        prod_id = [v for k, v in enumerate(product.keys())][0]
        
        for review in reviews[i][prod_id]: # iterate through each review
            review_id = review['review_id']
            for asp in review['aspect_pairs']:
                noun = asp['noun']
                adj = asp['adj']
                rule = asp['rule']
                polarity_nltk = asp['polarity_nltk']
                polarity_textblob = asp['polarity_textblob']
                cluster = asp['cluster']
                new_row = pd.DataFrame({'product_id':[prod_id], 'review_id':[review_id], 
                                        'noun':[noun], 'adj':[adj], 'rule':[rule], 
                                        'polarity_nltk':[polarity_nltk], 
                                        'polarity_textblob':[polarity_textblob],
                                        'cluster':[cluster]})
                tweet_clustered = tweet_clustered.append(new_row, ignore_index=True)
    return tweet_clustered