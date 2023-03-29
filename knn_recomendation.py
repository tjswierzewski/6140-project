from sklearn.model_selection import train_test_split
from create_numpy_from_data import Song, SongList
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as ssparse
from scoring import full_score
from multiprocessing import Pool
from functools import partial

NEIGHBORS = 200
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
        related_indices = model.kneighbors(data.T[index],return_distance = False)
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
    item_model = NearestNeighbors(n_neighbors = NEIGHBORS, metric='cosine', n_jobs=1)
    item_model.fit(train.T)

    # Create KNN user based model class
    user_model = NearestNeighbors(n_neighbors = NEIGHBORS, metric='cosine', n_jobs=1)
    user_model.fit(train)

    # Multiprocessing for playlist analysis
    pool = Pool()
    func = partial(get_playlist_recommendation_item_based, item_model, train)
    test_list = [i for i in test]
    scores = pool.map(func, test_list, 10)
    scores = [x for x in scores if x != ()]
    

    # Average Playlist Scores    
    npscores = np.array(scores)
    mean = np.mean(npscores, axis=0)
    print(f"Average:\nR_Percision: {mean[0]}\nNormalized Discounted Cumulative Gain: {mean[1]}\nRecommended Song Clicks: {mean[2]}")
if __name__ == "__main__":
    main()