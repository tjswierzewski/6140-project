from sklearn.model_selection import train_test_split
from create_numpy_from_data import Song, SongList
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as ssparse

NEIGHBORS = 200
TEST_LENGTH = 10
TEST_MIN = 15

def main():
    #Script Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--songlist", help = "Songlist Pickle", required= True)
    parser.add_argument("-m", "--matrix", help = "Data Matrix", required= True)
    parser.add_argument("-S", "--seed", help = "Seed", type= int, default= None)

    args = parser.parse_args()

    #Import SongList
    songlist = SongList()
    songlist.load(args.songlist)

    #Import Matrix
    matrix = ssparse.load_npz(args.matrix)


    #Split data into test and training
    train, test = train_test_split(matrix, test_size = .1, random_state=args.seed)

    #Create KNN model class
    model = NearestNeighbors(n_neighbors = NEIGHBORS, metric='cosine', n_jobs=-1)
    model.fit(train.T)


    for row in range(0, test.shape[0]):

        [_, song_index, playlist_position] = ssparse.find(test[row])
        if len(song_index) < TEST_MIN:
            print("Playlist too short\n")
            continue
        order = playlist_position.argsort()
        ordered_songs = song_index[order[:TEST_LENGTH]]

        raw_recommendations = {}

        for index in ordered_songs:

            # Return nearest neighbors
            related_indices = model.kneighbors(train.T[index],return_distance = False)
            for i, index in enumerate(related_indices[0]):
                if not index in ordered_songs:
                    if index in raw_recommendations.keys():
                        raw_recommendations[index] = (raw_recommendations[index][0] + i, raw_recommendations[index][1] + 1)
                    else:
                        raw_recommendations[index] = (i,1)
        
        recommendation_values = []
        for (position, count) in raw_recommendations.values():
            recommendation_values.append(position / count)
        recommendation_values = np.array(recommendation_values)
        recommended_songs = np.array(list(raw_recommendations.keys()))
        song_order = recommendation_values.argsort()[::-1]
        sorted_recommended = recommended_songs[song_order]
        union = np.intersect1d(sorted_recommended, song_index[TEST_LENGTH:])
        print(f"Length of playlist: {len(song_index)}\nNumber of matches: {len(union)}\n")




if __name__ == "__main__":
    main()