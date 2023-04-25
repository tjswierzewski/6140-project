from itertools import product
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from time import perf_counter_ns

from create_numpy_from_data import data_to_query_label, swap_song_index_to_X
from matrixfactor import computeFeatureVectors, getRecommendations
from scoring import full_score


query_length = [1, 10, 100]
test_sizes = [1, 1000]
features = [10,50,100,250,500]
reducer = "binary"

matrix = sparse.load_npz("UvS_sparse_matrix_D1000.npz")

print("--------MF Recommender--------")
for query, test_size, feature in product(query_length, test_sizes, features):
    print(f"\nQuery Length: {query} Test Length: {test_size} Latent Features: {feature}")
    train, test = train_test_split(matrix, test_size = test_size)
    test_queries, test_answers = data_to_query_label(test, query_length=query)
    Train = swap_song_index_to_X(train, shape=(train.shape[0], matrix.max()), reducer=reducer)
    Test = swap_song_index_to_X(test, shape=(test.shape[0], matrix.max()), reducer=reducer)
    compute_start = perf_counter_ns()
    playlist_features, song_features = computeFeatureVectors(Train, feature, 0.0001)
    compute_stop = perf_counter_ns()
    recommend_start = perf_counter_ns()
    recommendations = getRecommendations(Test, song_features, test_queries)
    recommend_stop = perf_counter_ns()
    scores = list(map(lambda given, recommended: full_score(given, recommended), test_answers, recommendations))
    np_scores = np.array(scores)
    mean = np.mean(np_scores, axis=0)
    counts = np.unique(np_scores, return_counts = True)
    no_click = counts[1][0] if counts[0][0] == 0 else 0
    no_find = counts[1][-1] if counts[0][-1] == 51 else 0
    print(f"\nItem Based Average:\nR_Precision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")
    print(f"No Click: {no_click}\nNo Find: {no_find}")
    print(f"Compute Time: {compute_stop- compute_start}\nRecommend Time: {recommend_stop - recommend_start}")