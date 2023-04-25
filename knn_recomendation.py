from sklearn.model_selection import train_test_split
from create_numpy_from_data import Song, SongList
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as ssparse
from scoring import full_score
from multiprocessing import Pool
from functools import partial
import pandas as pd
import scipy
from create_numpy_from_data import swap_song_index_to_X, data_to_query_label
from time import perf_counter_ns

ITEM_NEIGHBORS = 200
USER_NEIGHBORS = 20
TEST_LENGTH = 10
TEST_MIN = 20
REC_LENGTH = 500

def rank_merge(row, full_out = False):
    rank = {}
    def aggregate(list):
        for index, song_index in enumerate(list):
            if song_index not in rank.keys():
                rank[song_index] = 1 - (index / (len(list)))
            else:
                rank[song_index] += 1 - (index / (len(list)))
    row.map(aggregate)
    sorted_rank = sorted(rank.items(), key=lambda x:x[1], reverse=True)
    if full_out:
        return [x for x in sorted_rank][:500]    
    return [x[0] for x in sorted_rank][:500]

def remove_zeros(l):
    for ele in reversed(l):
        if not ele:
            del l[-1]
        else:
            break

def get_song_based_recommendations(data, query_playlists):
    query_playlists = [[x-1 for x in y]for y in query_playlists]
    model = NearestNeighbors(n_neighbors = ITEM_NEIGHBORS, metric='cosine', n_jobs=-1)
    fit_start = perf_counter_ns()
    model.fit(data)
    fit_stop = perf_counter_ns()
    song_list = np.unique(query_playlists)
    recommend_start = perf_counter_ns()
    recommendation_by_song = model.kneighbors(data[song_list],return_distance = False)
    recommend_stop = perf_counter_ns()
    reduce_start = perf_counter_ns()
    query_df =  pd.DataFrame(query_playlists).applymap(lambda x: recommendation_by_song[np.where(song_list == x)[0][0]])
    reduce_stop = perf_counter_ns()
    return query_df.apply(rank_merge, axis = 1).tolist(), fit_stop - fit_start, recommend_stop - recommend_start, reduce_stop - reduce_start
    
        
def get_playlist_recommendation_user_based(data, query_playlists):
    Train = swap_song_index_to_X(data)
    user_model = NearestNeighbors(n_neighbors = USER_NEIGHBORS, metric='cosine', n_jobs=-1)
    fit_start = perf_counter_ns()
    user_model.fit(Train)
    fit_stop = perf_counter_ns()
    query_playlists = ssparse.csr_matrix(query_playlists)
    query_playlists = swap_song_index_to_X(query_playlists, shape = (query_playlists.shape[0], Train.shape[1]))

    # Calculate nearest neighbors for playlist
    recommend_start = perf_counter_ns()
    related_indices = user_model.kneighbors(query_playlists ,return_distance = False)
    recommend_stop = perf_counter_ns()
    reduce_start = perf_counter_ns()
    query_df = pd.DataFrame(related_indices).applymap(lambda x: np.trim_zeros(data[x].toarray()[0])-1)
    reduce_stop = perf_counter_ns()
    return query_df.apply(rank_merge, axis = 1).tolist(), fit_stop - fit_start, recommend_stop - recommend_start, reduce_stop - reduce_start


def main():
    # Script Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--songlist", help = "Songlist Pickle", required= True)
    parser.add_argument("-m", "--matrix", help = "Data Matrix", required= True)
    parser.add_argument("-S", "--seed", help = "Seed", type= int, default= None)

    args = parser.parse_args()

    # Import SongList
    songlist = SongList()
    songlist.load(args.songlist)

    # Import Matrix
    matrix = ssparse.load_npz(args.matrix)

    # Split data into test and training
    train, test = train_test_split(matrix, test_size = .1, random_state=args.seed)
    Train = swap_song_index_to_X(train)    
    query_playlists, query_answers = data_to_query_label(test)

    # Get recommendations for songs in queries
    item_recommendations, fit_time, recommend_time, reduce_time = get_song_based_recommendations(Train.T, query_playlists)
    
    # Calculate item scores
    item_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, item_recommendations))
    
    # Get recommendations based on users
    user_recommendations, fit_time, recommend_time, reduce_time = get_playlist_recommendation_user_based(train, query_playlists)
    
    # Calculate item scores
    user_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, user_recommendations))

    # Average Playlist Scores    
    np_user_based_scores = np.array(user_based_scores)

    # Average Playlist Scores    
    np_item_based_scores = np.array(item_based_scores)
    mean = np.mean(np_item_based_scores, axis=0)
    print(f"\nItem Based Average:\nR_Precision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}\n")

    mean = np.mean(np_user_based_scores, axis=0)
    print(f"\nUser Based Average:\nR_Precision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}\n")

if __name__ == "__main__":
    main()