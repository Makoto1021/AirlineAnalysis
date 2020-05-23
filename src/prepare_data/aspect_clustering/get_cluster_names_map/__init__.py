from collections import defaultdict

def get_cluster_names_map(asp_to_cluster_map, aspect_freq_map):
    # assigns the most frequent noun for each cluster
    cluster_id_to_name_map = defaultdict()
    clusters = set(asp_to_cluster_map.values())
    for i in clusters:
        this_cluster_asp = [k for k,v in asp_to_cluster_map.items() if v == i]
        filt_freq_map = {k:v for k,v in aspect_freq_map.items() if k in this_cluster_asp}
        filt_freq_map = sorted(filt_freq_map.items(), key = lambda x: x[1], reverse = True)
        cluster_id_to_name_map[i] = filt_freq_map[0][0]
    return cluster_id_to_name_map