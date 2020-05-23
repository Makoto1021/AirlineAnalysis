from aspect_clustering.add_clusters_to_reviews import add_clusters_to_reviews
from aspect_clustering.aspect_to_pandas import aspect_to_pandas
from aspect_clustering.assign_groups import assign_groups

def update_reviews_data(reviews_data, nlp):
    updated_reviews = []
    ctr = 0
    print("Total number of unique products in this category: {}".format(len(reviews_data)))

    
    for i, product in enumerate(reviews_data):
        # print(i)
        prod_id = [v for k, v in enumerate(product.keys())][0]
        # this picks up one product and retrieves all the reviews about the product
        this_product_reviews = reviews_data[i][prod_id]
        print('prod_id is ', prod_id)
        # print("this product_reviews", this_product_reviews)
        # this picks up one product and retrieves all the reviews about the product
        # print('this_product_reviews is ', this_product_reviews)
        this_product_upd_reviews = add_clusters_to_reviews(this_product_reviews, prod_id, nlp)
        print("length of updated reviews:", len(this_product_upd_reviews[prod_id]))
        updated_reviews.append(this_product_upd_reviews)
        print("updated reviews added")
    reviews_clustered = aspect_to_pandas(updated_reviews)
    # reviews_clustered.to_csv('data/processed/reviews_clustered.csv', index=True, index_label=False)
    reviews_grouped = assign_groups(prod_id, reviews_clustered, nlp)
    # reviews_grouped.to_csv('data/processed/reviews_grouped.csv', index=True, index_label=False)
    
    return reviews_clustered, reviews_grouped