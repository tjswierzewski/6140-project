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
from create_numpy_from_data import swap_song_index_to_X

ITEM_NEIGHBORS = 200
USER_NEIGHBORS = 20
TEST_LENGTH = 10
TEST_MIN = 20
REC_LENGTH = 500

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, ssparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def data_to_query_label(data):
    query_playlists = data[:,0:10]
    query_answers = data[:,10:]
    rows_to_remove = np.where(np.unique(data.nonzero()[0], return_counts=True)[1] < TEST_MIN)[0]
    query_playlists = delete_rows_csr(query_playlists, rows_to_remove)
    query_answers = delete_rows_csr(query_answers, rows_to_remove)
    answers = query_answers.toarray().tolist()
    answers = [np.trim_zeros(x) for x in answers ]
    answers = [[x-1 for x in y]for y in answers]
    query_playlists = query_playlists.toarray().tolist()
    # query_playlists = [[x-1 for x in y]for y in query_playlists]
    return query_playlists, answers

def rank_merge(row):
    rank = {}
    def aggregate(list):
        for index, song_index in enumerate(list):
            if song_index not in rank.keys():
                rank[song_index] = 1 - (index / (len(list)))
            else:
                rank[song_index] += 1 - (index / (len(list)))
    row.map(aggregate)
    sorted_rank = sorted(rank.items(), key=lambda x:x[1], reverse=True)
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
    model.fit(data)
    song_list = np.unique(query_playlists)
    recommendation_by_song = model.kneighbors(data[song_list],return_distance = False)
    query_df =  pd.DataFrame(query_playlists).applymap(lambda x: recommendation_by_song[np.where(song_list == x)[0][0]])
    return query_df.apply(rank_merge, axis = 1).tolist()
    
        
def get_playlist_recommendation_user_based(data, query_playlists):
    Train = swap_song_index_to_X(data)
    user_model = NearestNeighbors(n_neighbors = USER_NEIGHBORS, metric='cosine', n_jobs=-1)
    user_model.fit(Train)
    query_playlists = ssparse.csr_matrix(query_playlists)
    query_playlists = swap_song_index_to_X(query_playlists, shape = (query_playlists.shape[0], Train.shape[1]))

    # Calculate nearest neighbors for playlist
    related_indices = user_model.kneighbors(query_playlists ,return_distance = False)
    query_df = pd.DataFrame(related_indices).applymap(lambda x: np.trim_zeros(data[x].toarray()[0])-1)
    return query_df.apply(rank_merge, axis = 1).tolist()


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
    item_recommendations = get_song_based_recommendations(Train.T, query_playlists)
    
    # Calculate item scores
    item_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, item_recommendations))
    
    # Get recommendations based on users
    user_recommendations = get_playlist_recommendation_user_based(train, query_playlists)
    
    # Calculate item scores
    user_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, user_recommendations))

    # Average Playlist Scores    
    np_user_based_scores = np.array(user_based_scores)

    # Average Playlist Scores    
    np_item_based_scores = np.array(item_based_scores)
    mean = np.mean(np_item_based_scores, axis=0)
    print(f"\nItem Based Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}\n")

    mean = np.mean(np_user_based_scores, axis=0)
    print(f"\nUser Based Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}\n")

if __name__ == "__main__":
    main()