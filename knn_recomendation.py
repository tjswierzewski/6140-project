from sklearn.model_selection import train_test_split
from create_numpy_from_data import Song, SongList
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as ssparse
from scoring import full_score
from multiprocessing import Pool
from functools import partial

ITEM_NEIGHBORS = 200
USER_NEIGHBORS = 20
TEST_LENGTH = 10
TEST_MIN = 15
REC_LENGTH = 500

def get_playlist_recommendation_item_based(model, data, playlist):
    [_, song_index, playlist_position] = ssparse.find(playlist)

    # Skip playlists that are too short to test
    if len(song_index) < TEST_MIN:
        return ()

    # Sort songs by playlist index and truncate to test length
    order = playlist_position.argsort()
    ordered_songs = song_index[order[:TEST_LENGTH]]

    # Create data structure to keep track of recommendation and rank
    raw_recommendations = {}
    for index in ordered_songs:

        # Calculate nearest neighbors for song
        related_indices = model.kneighbors(data[index],return_distance = False)
        for i, index in enumerate(related_indices[0]):
            # Skip songs in truncated playlist
            if not index in ordered_songs:
                if index in raw_recommendations.keys():
                    raw_recommendations[index] = raw_recommendations[index] + 500 - i
                else:
                    raw_recommendations[index] = 500-i

    # Order recommendations based on response for each song
    recommendation_values = []
    for position in raw_recommendations.values():
        recommendation_values.append(position / TEST_LENGTH)
    recommendation_values = np.array(recommendation_values)
    recommended_songs = np.array(list(raw_recommendations.keys()))
    song_order = recommendation_values.argsort()[::-1]
    sorted_recommended = recommended_songs[song_order][:REC_LENGTH]

    # Scoring
    return full_score(song_index[order[TEST_LENGTH:]], sorted_recommended)
        
def get_playlist_recommendation_user_based(model, data, playlist):
    [_, song_index, playlist_position] = ssparse.find(playlist)
    # Skip playlists that are too short to test
    if len(song_index) < TEST_MIN:
        return ()
    
    # Sort songs by playlist index and truncate to test length
    order = playlist_position.argsort()
    test_songs = song_index[order[TEST_LENGTH:]]
    given_playlist = ssparse.csr_matrix((playlist_position[order[:TEST_LENGTH]], (np.zeros(len(order[:TEST_LENGTH]), dtype=int),song_index[order[:TEST_LENGTH]])), shape = playlist.shape)

    # Calculate nearest neighbors for playlist
    related_indices = model.kneighbors(given_playlist ,return_distance = False)

    raw_recommendations = {}
    rec_playlists = data[related_indices[0]]
    sums = rec_playlists.sum(0)
    rec_song_count = rec_playlists.getnnz(0) * 500
    scores = (rec_song_count - sums) / USER_NEIGHBORS
    [_, rec_song, score] = ssparse.find(scores)
    rec_order = score.argsort()
    recommended_songs = rec_song[rec_order]
    recommended_songs = [x for x in recommended_songs if x not in song_index[order[:TEST_LENGTH]]]
    
    # Scoring
    return full_score(test_songs, recommended_songs)

        



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

    # Create KNN item based model class
    item_model = NearestNeighbors(n_neighbors = ITEM_NEIGHBORS, metric='cosine', n_jobs=1)
    item_model.fit(train.T)

    # Create KNN user based model class
    user_model = NearestNeighbors(n_neighbors = USER_NEIGHBORS, metric='cosine', n_jobs=1)
    user_model.fit(train)

    # Iterator for test playlists
    test_list = [i for i in test]

    # Multiprocessing for playlist analysis
    pool = Pool()
    func = partial(get_playlist_recommendation_user_based, user_model, train)
    user_based_scores = pool.map(func, test_list, 10)
    user_based_scores = [x for x in user_based_scores if x != ()]


    # Multiprocessing for playlist analysis
    pool = Pool()
    func = partial(get_playlist_recommendation_item_based, item_model, train.T)
    item_based_scores = pool.map(func, test_list, 10)
    item_based_scores = [x for x in item_based_scores if x != ()]
    
    # Average Playlist Scores    
    np_user_based_scores = np.array(user_based_scores)
    mean = np.mean(np_user_based_scores, axis=0)
    print("")
    print(f"User Based Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")
    print("")
    # Average Playlist Scores    
    np_item_based_scores = np.array(item_based_scores)
    mean = np.mean(np_item_based_scores, axis=0)
    print(f"Item Based Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")


if __name__ == "__main__":
    main()