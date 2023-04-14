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
    query_playlists = [[x-1 for x in y]for y in query_playlists]
    return query_playlists, answers


def get_song_based_recommendations(data, query_playlists):
    model = NearestNeighbors(n_neighbors = ITEM_NEIGHBORS, metric='cosine', n_jobs=-1)
    model.fit(data)
    song_list = np.unique(query_playlists)
    recommendation_by_song = model.kneighbors(data[song_list],return_distance = False)
    query_df =  pd.DataFrame(query_playlists).applymap(lambda x: recommendation_by_song[np.where(song_list == x)[0][0]])
    return query_df.apply(rank_merge, axis = 1).tolist()
    
        
def get_playlist_recommendation_user_based(data, playlist):
    user_model = NearestNeighbors(n_neighbors = USER_NEIGHBORS, metric='cosine', n_jobs=-1)
    user_model.fit(data)
    # [_, song_index, playlist_position] = ssparse.find(playlist)
    # # Skip playlists that are too short to test
    # if len(song_index) < TEST_MIN:
    #     return ()
    
    # # Sort songs by playlist index and truncate to test length
    # order = playlist_position.argsort()
    # test_songs = song_index[order[TEST_LENGTH:]]
    # given_playlist = ssparse.csr_matrix((playlist_position[order[:TEST_LENGTH]], (np.zeros(len(order[:TEST_LENGTH]), dtype=int),song_index[order[:TEST_LENGTH]])), shape = playlist.shape)

    # # Calculate nearest neighbors for playlist
    # related_indices = model.kneighbors(given_playlist ,return_distance = False)

    # raw_recommendations = {}
    # rec_playlists = data[related_indices[0]]
    # sums = rec_playlists.sum(0)
    # rec_song_count = rec_playlists.getnnz(0) * 500
    # scores = (rec_song_count - sums) / USER_NEIGHBORS
    # [_, rec_song, score] = ssparse.find(scores)
    # rec_order = score.argsort()
    # recommended_songs = rec_song[rec_order]
    # recommended_songs = [x for x in recommended_songs if x not in song_index[order[:TEST_LENGTH]]]
    
    # Scoring
    return full_score(test_songs, recommended_songs)


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

def swap_axes(matrix):
    [playlist_index, X_index, value] = ssparse.find(matrix)
    swapped_matrix = ssparse.csr_matrix((X_index + 1, (playlist_index, value-1)))
    # Normalize before summing duplicates
    recips = np.reciprocal(swapped_matrix.max(axis=1).toarray().astype(np.float32))
    recips = ssparse.csr_matrix(recips)
    swapped_matrix = recips.multiply(swapped_matrix).tocsr() * -1
    swapped_matrix[swapped_matrix != 0] += 1
    swapped_matrix.sum_duplicates()
    return swapped_matrix

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

    Train = swap_axes(train)    

    # Get recommendations for songs in queries
    query_playlists, query_answers = data_to_query_label(test)
    item_recommendations = get_song_based_recommendations(Train.T, query_playlists)
    
    # Calculate item scores
    item_based_scores = list(map(lambda given, recommended: full_score(given, recommended), query_answers, item_recommendations))
    
    # Get recommendations based on users

    get_playlist_recommendation_user_based(Train, query_playlists)
    
    # # Average Playlist Scores    
    # np_user_based_scores = np.array(user_based_scores)
    # mean = np.mean(np_user_based_scores, axis=0)
    # print("")
    # print(f"User Based Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")
    # print("")

    # Average Playlist Scores    
    np_item_based_scores = np.array(item_based_scores)
    mean = np.mean(np_item_based_scores, axis=0)
    print(f"\nItem Based Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}\n")


if __name__ == "__main__":
    main()