import numpy as np

def get_score(prob, alpha=6.0):
    mean_prob = 0.44
    std_prob = 0.22
    z = (prob - mean_prob) / std_prob
    sigmoid = 1 / (1 + np.exp(-z))
    a_score = alpha * sigmoid
    return round(a_score, 2)
