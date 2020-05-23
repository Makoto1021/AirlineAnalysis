from mapper import map

def get_aspects(reviews_data):
    # it retrieves the nouns in the reviews from a single product
    aspects = []
    for review in reviews_data:
        aspect_pairs = review["aspect_pairs"]
        for map in aspect_pairs:
            aspects.append(map['noun'])
    # print("aspects from get_aspects", aspects)
    return aspects