import json
import numpy

class Song:
    def __init__(self, name, track_uri) -> None:
        self.name = name
        self.uri = track_uri

    @classmethod
    def from_MFD(cls, song):
        return cls(song["track_name"], song["track_uri"].split(":")[2])



def main(depth = 1, path = ".."):
    start = 0
    stop = 999
    step = 1000
    songs = dict()
    for _ in range(depth):
        f = open(path + f"/spotify_million_playlist_dataset/data/mpd.slice.{start}-{stop}.json")
        data = json.load(f)
    
        for playlist in data["playlists"]:
            for song in playlist["tracks"]:
                object = Song.from_MFD(song)
                songs[object.uri] =  object
    
        start = start + step
        stop = stop + step
        f.close()   

    ordered_songs = list(songs.values())
    start = 0
    stop = 999
    step = 1000
    df = numpy.zeros((1,len(ordered_songs)))
    for _ in range(depth):
        f = open(path + f"/spotify_million_playlist_dataset/data/mpd.slice.{start}-{stop}.json")
        data = json.load(f)
    
        for playlist in data["playlists"]:
            row = numpy.zeros(len(ordered_songs))
            for song in playlist["tracks"]:
                object = Song.from_MFD(song)
                search_object = songs[object.uri]
                index = ordered_songs.index(search_object)
                row[index] = 1
            df = numpy.vstack([df,row])
        start = start + step
        stop = stop + step
        f.close()
    df = numpy.delete(df,0,0)

    return df


if(__name__ == "__main__"):
    main()
