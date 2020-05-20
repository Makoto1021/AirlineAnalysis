from aspect_clustering.get_word_vectors import get_word_vectors
import dataiku
from sklearn import cluster
import ast

n_clusters = dataiku.get_custom_variables(typed=True)['NUM_CLUSTERS']
# print(n_clusters)
n_clusters = ast.literal_eval(n_clusters)


def get_word_clusters(unique_aspects, nlp):
    
    print("Found {} unique aspects for this product".format(len(unique_aspects)))
    asp_vectors = get_word_vectors(unique_aspects, nlp) # gets the word vector for each noun 
    print("len(unique_aspects)", len(unique_aspects))
    print("n_clusters", n_clusters)
    if len(unique_aspects) < n_clusters:
        print("Too few aspects ({}) found. No clustering required...".format(len(unique_aspects)))
        return list(range(len(unique_aspects)))

    print("Running k-means clustering...")
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(asp_vectors)
    labels = kmeans.labels_
    return labels