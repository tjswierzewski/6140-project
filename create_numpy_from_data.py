import argparse
import bisect
import json
import pickle
import re
import numpy as np
import scipy.sparse as ssparse

class Song:
    def __init__(self, name, track_uri) -> None:
        self.name = name
        self.uri = track_uri

    @staticmethod
    def truncate_uri(uri):
        return uri.split(":")[2]
    
    @classmethod
    def from_MFD(cls, song):
        return cls(song["track_name"], cls.truncate_uri(song["track_uri"]))
    
    def __eq__(self, __o: object) -> bool:
        return self.uri == __o.uri
    
    def __lt__(self, __o: object) -> bool:
        return self.uri < __o.uri

    def __le__(self, __o: object) -> bool:
        return self.uri <= __o.uri

    def __gt__(self, __o: object) -> bool:
        return self.uri > __o.uri

    def __ge__(self, __o: object) -> bool:
        return self.uri >= __o.uri

    
    
class SongList:
    def __init__(self) -> None:
        self.list = []
    
    def insert(self, song):
        bisect.insort_left(self.list, song)

    def search(self, song_uri):
        index = bisect.bisect_left(self.list, song_uri, key = lambda x: x.uri)
        if len(self.list) != 0 and index != len(self.list) and self.list[index].uri == song_uri:
            return index
        return None
        
    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.list, file)

    def load(self, filename):
        with open(filename, "rb") as file:
            self.list = pickle.load(file)


def create_song_list(depth = 1, path = ".."):
    start = 0
    stop = 999
    step = 1000
    songs = SongList()
    for _ in range(depth):
        f = open(path + f"/spotify_million_playlist_dataset/data/mpd.slice.{start}-{stop}.json")
        data = json.load(f)
    
        for playlist in data["playlists"]:
            for song in playlist["tracks"]:
                if not songs.search(Song.truncate_uri(song["track_uri"])):
                    object = Song.from_MFD(song)
                    songs.insert(object)
    
        start = start + step
        stop = stop + step
        f.close()   
    songs.save(f"songlist_D{depth}.pickle")
    return songs

def create_user_song_dataframe(songlist, depth, path):
    start = 0
    stop = 999
    step = 1000
    rows = []
    playlist_index = []
    song_index = []
    position_index = []
    for _ in range(depth):
        f = open(path + f"/spotify_million_playlist_dataset/data/mpd.slice.{start}-{stop}.json")
        data = json.load(f)
    
        for playlist in data["playlists"]:
            row = np.zeros(len(songlist.list))
            for i , song in enumerate(playlist["tracks"]):
                index = songlist.search(Song.truncate_uri(song["track_uri"]))
                playlist_index.append(int(playlist['pid']))
                song_index.append(index + 1)
                position_index.append(i)
        start = start + step
        stop = stop + step
        f.close()
    df = ssparse.csr_matrix((song_index, (playlist_index, position_index)))

    print(df.shape)
    ssparse.save_npz(f"UvS_sparse_matrix_D{depth}", df)


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


def swap_song_index_to_X(matrix, shape = None, value='linear'):
    [playlist_index, X_index, value] = ssparse.find(matrix)
    swapped_matrix = ssparse.csr_matrix((X_index + 1, (playlist_index, value-1)), shape=shape)
    if value == 'linear':
        # Normalize before summing duplicates
        recips = np.reciprocal(swapped_matrix.max(axis=1).toarray().astype(np.float32))
        recips = ssparse.csr_matrix(recips)
        swapped_matrix = recips.multiply(swapped_matrix).tocsr() * -1
        swapped_matrix[swapped_matrix < -1] = -1
        swapped_matrix[swapped_matrix != 0] += 1
        swapped_matrix.sum_duplicates()
    elif value == 'binary':
        swapped_matrix.sum_duplicates()
        swapped_matrix[swapped_matrix != 0] = 1
    return swapped_matrix

def data_to_query_label(data, query_length = 20):
    query_playlists = data[:,0:10]
    query_answers = data[:,10:]
    rows_to_remove = np.where(np.unique(data.nonzero()[0], return_counts=True)[1] < query_length)[0]
    query_playlists = delete_rows_csr(query_playlists, rows_to_remove)
    query_answers = delete_rows_csr(query_answers, rows_to_remove)
    answers = query_answers.toarray().tolist()
    answers = [np.trim_zeros(x) for x in answers ]
    answers = [[x-1 for x in y]for y in answers]
    query_playlists = query_playlists.toarray().tolist()
    return query_playlists, answers

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help = "MFD folder path", default= "..")
    parser.add_argument("-d", "--depth", help = "Number of Slices to Use", type = int, default = 1)
    parser.add_argument("-P", "--pickle", help = "Pickle File")

    args = parser.parse_args()
    path = args.path
    depth = int(args.depth)
    song_pickle = args.pickle

    song_pickle

    if not song_pickle:
        songs = create_song_list(depth = depth, path = path)
    else:
        songs = SongList()
        songs.load(song_pickle)
        depth = int(re.search(r'\d+', song_pickle).group())

    create_user_song_dataframe(songs, depth, path)

if(__name__ == "__main__"):
    main()
