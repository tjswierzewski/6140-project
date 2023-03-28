import math
import numpy as np
import sklearn

def r_precision(truncated_playlist, recommendations, intersect = None):
    if type(intersect) == 'NoneType':
        intersect = np.intersect1d(truncated_playlist, recommendations)
    return len(intersect)/len(truncated_playlist)

def normalized_discounted_cumulative_gain(truncated_playlist, recommendations, intersect = None):
    if type(intersect) == 'NoneType':
        intersect = np.intersect1d(truncated_playlist, recommendations)
    if len(intersect) == 0:
        return 0
    dcg = 0
    for i, rel in enumerate(np.isin(recommendations, truncated_playlist).astype(int)):
        if i == 0:
            dcg += rel
        else:
            dcg += rel/math.log2(i+1)
    idcg = 0
    for i in range(len(intersect)):
        if i == 0:
            idcg += 1
        else:
            idcg += 1/math.log2(i+1)
    return dcg / idcg


def recommended_song_clicks(truncated_playlist, recommendations):
    song = truncated_playlist[0]
    index = np.where(recommendations == song)
    if index[0].size == 0:
        return 51
    return int(index[0][0]) // 10


def full_score(truncated_playlist, recommendations):
    intersect = np.intersect1d(truncated_playlist, recommendations)
    r_p = r_precision(truncated_playlist, recommendations, intersect)
    ndcg = normalized_discounted_cumulative_gain(truncated_playlist, recommendations, intersect)
    rsc = recommended_song_clicks(truncated_playlist, recommendations)
    return [r_p, ndcg, rsc]