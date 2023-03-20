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
    
    
class SongList:
    def __init__(self) -> None:
        self.list = []
    
    def binary_search(self,item_uri):
        low = 0
        high = len(self.list) - 1
        mid = 0
        check = hash(item_uri)

        while low <= high:

            mid = high + low // 2

            if self.list[mid].hash < check:
                low = mid + 1
            elif self.list[mid].hash > check:
                high = mid - 1
            else:
                return mid
        return mid - 1
    
    def insert(self, song):
        if len(self.list) != 0: 
            index = self.binary_search(song)
            if song.hash == self.list[index].hash:
                return
        else:
            index = 0
        self.list.insert(index, song)

    def search(self, song_uri):
        if len(self.list) == 0:
            return None
        index = self.binary_search(song_uri)
        song = self.list[index]
        if song.hash == hash(song_uri):
            return song
        return None


def main(depth = 1, path = ".."):
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
