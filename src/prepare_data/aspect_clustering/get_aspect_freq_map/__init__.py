from collections import defaultdict

def get_aspect_freq_map(aspects):
    # simply counts how many times each noun appears
    aspect_freq_map = defaultdict(int)
    for asp in aspects:
        aspect_freq_map[asp] += 1
    return aspect_freq_map