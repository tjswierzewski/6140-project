from sklearn.model_selection import train_test_split
from create_numpy_from_data import Song, SongList
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as ssparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--songlist", help = "Songlist Pickle", required= True)
    parser.add_argument("-m", "--matrix", help = "Data Matrix", required= True)
    parser.add_argument("-S", "--seed", help = "Seed", type= int, default= None)

    args = parser.parse_args()

    songlist = SongList()
    songlist.load(args.songlist)

    matrix = ssparse.load_npz(args.matrix)

    train, test = train_test_split(matrix, test_size = .1, random_state=args.seed)

    model = NearestNeighbors(n_neighbors = 100, metric='cosine', n_jobs=-1)

    model.fit(train.T)

    input = train.nonzero()[1][0]

    indices = model.kneighbors(train.T[input],return_distance = False)

    print(f"Input song is {songlist.list[input].name}")

    for i in indices[0][0:10]:
        print(songlist.list[i].name)

    union = np.intersect1d(test[0].nonzero()[1], indices[0])
    print(f"Number of common terms: {len(union)}")





if __name__ == "__main__":
    main()