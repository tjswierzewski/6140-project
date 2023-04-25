from itertools import product
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

from create_numpy_from_data import data_to_query_label, swap_song_index_to_X
from knn_recomendation import get_playlist_recommendation_user_based, get_song_based_recommendations
from scoring import full_score


query_length = [1, 10, 100]
test_sizes = [1, 1000]
reducer = "binary"

matrix = sparse.load_npz("UvS_sparse_matrix_D1000.npz")
f = open("knn_out.txt", "w")

f.write("--------KNN Recommender--------")
for query, test_size in product(query_length, test_sizes):
    f.write(f"\nQuery Length: {query} Test Length: {test_size}")
    train, test = train_test_split(matrix, test_size = 0.01)
    query_playlists, query_answers = data_to_query_label(test, query_length=query, max_return=test_size)
    Train = swap_song_index_to_X(train, shape=(matrix.shape[0], matrix.max()), reducer=reducer)

    item_recommendations, fit_time, recommend_time, reduce_time = get_song_based_recommendations(Train.T, query_playlists)

    item_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, item_recommendations))
    np_item_based_scores = np.array(item_based_scores)
    mean = np.mean(np_item_based_scores, axis=0)
    counts = np.unique(np_item_based_scores, return_counts = True)
    no_click = counts[1][0] if counts[0][0] == 0 else 0
    no_find = counts[1][-1] if counts[0][-1] == 51 else 0
    f.write(f"\nN = {len(query_playlists)}")
    f.write(f"Item Based Average:\nR_Precision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")
    f.write(f"No Click: {no_click}\nNo Find: {no_find}")
    f.write(f"Fit Time: {fit_time}\nRecommend Time: {recommend_time}\nReduce Time: {reduce_time}")

    train, test = train_test_split(matrix, test_size = test_size)
    Train = swap_song_index_to_X(train, shape=(matrix.shape[0], matrix.max()), reducer=reducer)
    query_playlists, query_answers = data_to_query_label(test, query_length=query)
    
    user_recommendations, fit_time, recommend_time, reduce_time = get_playlist_recommendation_user_based(train, query_playlists)

    user_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, user_recommendations))
    np_user_based_scores = np.array(user_based_scores)
    mean = np.mean(np_user_based_scores, axis=0)
    counts = np.unique(np_item_based_scores, return_counts = True)
    no_click = counts[1][0] if counts[0][0] == 0 else 0
    no_find = counts[1][-1] if counts[0][-1] == 51 else 0
    f.write(f"\nN = {len(query_playlists)}")
    f.write(f"User Based Average:\nR_Precision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")
    f.write(f"No Click: {no_click}\nNo Find: {no_find}")
    f.write(f"Fit Time: {fit_time}\nRecommend Time: {recommend_time}\nReduce Time: {reduce_time}")