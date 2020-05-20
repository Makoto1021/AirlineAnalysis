import numpy as np

def vector_dist(vec1, vec2):
    dist = np.linalg.norm(vec1-vec2)
    return dist