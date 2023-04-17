import argparse
import pickle
from scipy import sparse
from sklearn.model_selection import train_test_split
from create_numpy_from_data import SongList, Song, data_to_query_label, swap_song_index_to_X
from matrixfactor import computeFeatureVectors, predictPlaylistFeatures


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

    matrix = sparse.load_npz(args.matrix)

    train, test = train_test_split(matrix, test_size=0.1, random_state=args.seed)
    train, validate = train_test_split(train, test_size=0.25, random_state=args.seed)
    test, keys = data_to_query_label(test)

    output = {}
    train_playlist_features, train_song_features = computeFeatureVectors(swap_song_index_to_X(train, shape=(train.shape[0], matrix.max())), 70, 0.0001)
    output["train_playlist_features"] = train_playlist_features
    output["train_song_features"] = train_song_features
    output["validate_playlist_features"] = predictPlaylistFeatures(swap_song_index_to_X(validate, shape=(validate.shape[0], matrix.max())), train_song_features)
    output["test_playlist_features"] = predictPlaylistFeatures(swap_song_index_to_X(test, shape=(len(test), matrix.max())), train_song_features)
    output["keys"] = keys
    output["train"] = train
    output["validate"] = validate

    with open("NN_data_D1.pickle", "wb") as file:
            pickle.dump(output, file)


if __name__ == "__main__":
    main()