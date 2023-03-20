import bisect
import json
import numpy

class Song:
    def __init__(self, name, track_uri) -> None:
        self.name = name
        self.uri = track_uri
        self.hash = hash(track_uri)

    @staticmethod
    def truncate_uri(uri):
        return uri.split(":")[2]
    
    @classmethod
    def from_MFD(cls, song):
        return cls(song["track_name"], cls.truncate_uri(song["track_uri"]))
    
    def __eq__(self, __o: object) -> bool:
        return self.hash == __o.hash
    
    def __lt__(self, __o: object) -> bool:
        return self.hash < __o.hash

    def __le__(self, __o: object) -> bool:
        return self.hash <= __o.hash

    def __gt__(self, __o: object) -> bool:
        return self.hash > __o.hash

    def __ge__(self, __o: object) -> bool:
        return self.hash >= __o.hash

    
    
class SongList:
    def __init__(self) -> None:
        self.list = []
    
    def insert(self, song):
        bisect.insort_left(self.list, song)

    def search(self, song_uri):
        index = bisect.bisect_left(self.list, hash(song_uri), key = lambda x: x.hash)
        if len(self.list) != 0 and index != len(self.list) and self.list[index].uri == song_uri:
            return index
        return None
        

def main(depth = 5, path = ".."):
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

    
    start = 0
    stop = 999
    step = 1000
    rows = []
    for _ in range(depth):
        f = open(path + f"/spotify_million_playlist_dataset/data/mpd.slice.{start}-{stop}.json")
        data = json.load(f)
    
        for playlist in data["playlists"]:
            row = numpy.zeros(len(songs.list))
            for song in playlist["tracks"]:
                index = songs.search(song["track_uri"])
                row[index] = 1
            rows.append([row])
        start = start + step
        stop = stop + step
        f.close()
    df = numpy.concatenate(rows)

    print(df.shape)
    numpy.save(f"UvS_matrix_D{depth}", df)


if(__name__ == "__main__"):
    main()
