from aspect_clustering.get_aspects import get_aspects
from aspect_clustering.get_aspect_freq_map import get_aspect_freq_map
from aspect_clustering.get_word_clusters import get_word_clusters
from aspect_clustering.get_cluster_names_map import get_cluster_names_map

def add_clusters_to_reviews(reviews_data, prod_id, nlp):
    product_aspects = get_aspects(reviews_data) # all the nouns from the reviews in a single product
    aspect_freq_map = get_aspect_freq_map(product_aspects) # counts the frequency of each noun in a single product
    unique_aspects = aspect_freq_map.keys() # gets the unique nouns

    aspect_labels = get_word_clusters(unique_aspects, nlp)
    asp_to_cluster_map = dict(zip(unique_aspects, aspect_labels))
    cluster_names_map = get_cluster_names_map(asp_to_cluster_map, aspect_freq_map)
    updated_reviews = []

    for review in reviews_data: # picks up one review
        aspect_pairs_upd = []
        aspect_pairs = review["aspect_pairs"]
        for map in aspect_pairs:
            noun = map['noun'] # picks up a noun
            cluster_label_id = asp_to_cluster_map[noun] # assigns the cluster 
            cluster_label_name = cluster_names_map[cluster_label_id] # get's the most frequent noun for the assigned cluster
            map['cluster'] = cluster_label_name
            aspect_pairs_upd.append(map)

        review['aspect_pairs'] = aspect_pairs_upd
        updated_reviews.append(review)
    result = {prod_id:updated_reviews}
    return result