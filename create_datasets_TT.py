import argparse
import pickle
from scipy import sparse
from sklearn.model_selection import train_test_split
from create_numpy_from_data import SongList, Song, data_to_query_label, swap_song_index_to_X
from matrixfactor import computeFeatureVectors, predictPlaylistFeatures


def main():
    # Script Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--matrix", help = "Data Matrix", required= True)
    parser.add_argument("-S", "--seed", help = "Seed", type= int, default= None)
    parser.add_argument("-v", "--validation", help = "size of training data used for validation", type=float)
    parser.add_argument("-f", "--features", help = "Number of features in MF", type=int, default=100)

    args = parser.parse_args()

    matrix = sparse.load_npz(args.matrix)

    train, test = train_test_split(matrix, test_size=0.1, random_state=args.seed)
    if args.validation != None:
        train, validate = train_test_split(train, test_size=args.validation, random_state=args.seed)
    test, keys = data_to_query_label(test)

    output = {}
    train_playlist_features, train_song_features = computeFeatureVectors(swap_song_index_to_X(train, shape=(train.shape[0], matrix.max())), args.features, 0.0001)
    output["train_playlist_features"] = train_playlist_features
    output["train_song_features"] = train_song_features
    output["train"] = train
    if args.validation != None:
        output["validate_playlist_features"] = predictPlaylistFeatures(swap_song_index_to_X(validate, shape=(validate.shape[0], matrix.max())), train_song_features)
        output["validate"] = validate
    output["test_playlist_features"] = predictPlaylistFeatures(swap_song_index_to_X(test, shape=(len(test), matrix.max())), train_song_features)
    output["test"] = test
    output["keys"] = keys

    file_name = args.matrix.split("_")[-1].split(".")[0] + f"_F{args.features}"
    if args.validation:
        file_name += "_V"
    file_name += ".pickle"
    with open(file_name, "wb") as file:
            pickle.dump(output, file)


if __name__ == "__main__":
    main()