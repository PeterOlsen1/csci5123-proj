import json
import random

def create_challenge_set():
    '''
    This function creates a mock challenge set in accordance with what was
    outlined within our paper.

    Call this module in the top level directory of the project to create
    the test data challenge set. There must be a spotify MPD slice in the same
    directory named "challenge_set_data.json".

    (1) title only (2) title and the first track (3) title and the first 
    five tracks (4) the first five tracks (no title) 
    (5) title and the first 10 tracks (6) the first 10 tracks (no title) 
    (7) title and the first 25 tracks (8) title and 25 random tracks
    (9) title and the first 100 tracks (10) title and 100 random tracks. A
    final submission for this challenge should contain 500 tracks for
    each of the test playlists, ordered by relevance.
    '''
    fp = open("challenge_set_data.json", "r")
    data = json.load(fp)
    fp.close()

    fp = open("challenge_set.json", "w")
    write_data = []
    written_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, playlist in enumerate(data['playlists']):
        if playlist['num_tracks'] > 100 and written_data[9] < 100:
            # title + random 100 tracks
            playlist['num_samples'] = 100
            new_tracks = random.sample(playlist['tracks'], 100)
            playlist['tracks'] = new_tracks
            playlist['num_holdouts'] = playlist['num_tracks'] - 100

            write_data.append(playlist)
            written_data[9] += 1
        elif playlist['num_tracks'] > 100 and written_data[8] < 100:
            # title + first 100 tracks
            playlist['tracks'] = playlist['tracks'][:100]
            playlist['num_holdouts'] = playlist['num_tracks'] - 100
            playlist['num_samples'] = 0
            write_data.append(playlist)
            written_data[8] += 1
        elif playlist['num_tracks'] > 25 and written_data[7] < 100:
            # title + random 25 tracks
            playlist['num_samples'] = 25
            new_tracks = random.sample(playlist['tracks'], 25)
            playlist['tracks'] = new_tracks
            playlist['num_holdouts'] = playlist['num_tracks'] - 25

            write_data.append(playlist)
            written_data[7] += 1
        elif playlist['num_tracks'] > 25 and written_data[6] < 100:
            # title + first 25 tracks
            playlist['tracks'] = playlist['tracks'][:25]
            playlist['num_holdouts'] = playlist['num_tracks'] - 25
            playlist['num_samples'] = 0
            write_data.append(playlist)
            written_data[6] += 1
        elif playlist['num_tracks'] > 10 and written_data[5] < 100:
            # no title + first 10 tracks
            playlist['name'] = ""
            playlist['tracks'] = playlist['tracks'][:10]
            playlist['num_holdouts'] = playlist['num_tracks'] - 10
            playlist['num_samples'] = 0
            write_data.append(playlist)
            written_data[5] += 1
        elif playlist['num_tracks'] > 10 and written_data[4] < 100:
            # title + first 10 tracks
            playlist['tracks'] = playlist['tracks'][:10]
            playlist['num_holdouts'] = playlist['num_tracks'] - 10
            playlist['num_samples'] = 10
            write_data.append(playlist)
            written_data[4] += 1
        elif playlist['num_tracks'] > 5 and written_data[3] < 100:
            # no title + first 5 tracks
            playlist['name'] = ""
            playlist['tracks'] = playlist['tracks'][:5]
            playlist['num_holdouts'] = playlist['num_tracks'] - 5
            playlist['num_samples'] = 5
            write_data.append(playlist)
            written_data[3] += 1
        elif playlist['num_tracks'] > 5 and written_data[2] < 100:
            # title + first 5 tracks
            playlist['tracks'] = playlist['tracks'][:5]
            playlist['num_holdouts'] = playlist['num_tracks'] - 5
            playlist['num_samples'] = 5
            write_data.append(playlist)
            written_data[2] += 1
        elif playlist['num_tracks'] > 1 and written_data[1] < 100:
            # title + first track
            playlist['tracks'] = [playlist['tracks'][0]]
            playlist['num_holdouts'] = playlist['num_tracks'] - 1
            playlist['num_samples'] = 1
            write_data.append(playlist)
            written_data[1] += 1
        else:
            # title only
            playlist['tracks'] = []
            playlist['num_holdouts'] = playlist['num_tracks']
            playlist['num_samples'] = 0
            write_data.append(playlist)

    data['playlists'] = write_data
    json.dump(data, fp, indent=4)
    fp.close()


create_challenge_set()