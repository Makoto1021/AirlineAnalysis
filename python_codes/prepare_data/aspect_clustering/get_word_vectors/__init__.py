def get_word_vectors(unique_aspects, nlp):
    asp_vectors = []
    for aspect in unique_aspects:
        token = nlp(aspect)
        asp_vectors.append(token.vector)
    return asp_vectors