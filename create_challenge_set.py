import json
import random
import copy

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
    data_copy = copy.deepcopy(data)
    fp.close()

    fp = open("challenge_set.json", "w")
    answers = open("challenge_set_answers.json", "w")
    write_data = []
    answer_data = []
    written_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, playlist in enumerate(data['playlists']):
        playlist["pid"] = i
        answer = copy.deepcopy(playlist)
        random_flag = True

        if playlist['num_tracks'] > 100 and written_data[9] < 100:
            # title + random 100 tracks
            playlist['num_samples'] = 100
            new_tracks = random.sample(playlist['tracks'], 100)
            playlist['tracks'] = new_tracks
            playlist['num_holdouts'] = playlist['num_tracks'] - 100

            written_data[9] += 1
        elif playlist['num_tracks'] > 100 and written_data[8] < 100:
            # title + first 100 tracks
            playlist['tracks'] = playlist['tracks'][:100]
            playlist['num_holdouts'] = playlist['num_tracks'] - 100
            playlist['num_samples'] = 100
            written_data[8] += 1
            random_flag = False

        elif playlist['num_tracks'] > 25 and written_data[7] < 100:
            # title + random 25 tracks
            playlist['num_samples'] = 25
            new_tracks = random.sample(playlist['tracks'], 25)
            playlist['tracks'] = new_tracks
            playlist['num_holdouts'] = playlist['num_tracks'] - 25
            written_data[7] += 1

        elif playlist['num_tracks'] > 25 and written_data[6] < 100:
            # title + first 25 tracks
            playlist['tracks'] = playlist['tracks'][:25]
            playlist['num_holdouts'] = playlist['num_tracks'] - 25
            playlist['num_samples'] = 25
            written_data[6] += 1
            random_flag = False

        elif playlist['num_tracks'] > 10 and written_data[5] < 100:
            # no title + first 10 tracks
            playlist['name'] = ""
            playlist['tracks'] = playlist['tracks'][:10]
            playlist['num_holdouts'] = playlist['num_tracks'] - 10
            playlist['num_samples'] = 10
            written_data[5] += 1
            random_flag = False

        elif playlist['num_tracks'] > 10 and written_data[4] < 100:
            # title + first 10 tracks
            playlist['tracks'] = playlist['tracks'][:10]
            playlist['num_holdouts'] = playlist['num_tracks'] - 10
            playlist['num_samples'] = 10
            written_data[4] += 1
            random_flag = False

        elif playlist['num_tracks'] > 5 and written_data[3] < 100:
            # no title + first 5 tracks
            playlist['name'] = ""
            playlist['tracks'] = playlist['tracks'][:5]
            playlist['num_holdouts'] = playlist['num_tracks'] - 5
            playlist['num_samples'] = 5
            written_data[3] += 1
            random_flag = False

        elif playlist['num_tracks'] > 5 and written_data[2] < 100:
            # title + first 5 tracks
            playlist['tracks'] = playlist['tracks'][:5]
            playlist['num_holdouts'] = playlist['num_tracks'] - 5
            playlist['num_samples'] = 5
            written_data[2] += 1
            random_flag = False

        elif playlist['num_tracks'] > 1 and written_data[1] < 100:
            # title + first track
            playlist['tracks'] = [playlist['tracks'][0]]
            playlist['num_holdouts'] = playlist['num_tracks'] - 1
            playlist['num_samples'] = 1
            written_data[1] += 1
            random_flag = False

        else:
            # title only
            playlist['tracks'] = []
            playlist['num_holdouts'] = playlist['num_tracks']
            playlist['num_samples'] = 0
            random_flag = False

        # for answer, "num_holdouts" will be the length of the "tracks" list
        answer['num_holdouts'] = playlist['num_holdouts']
        answer['num_samples'] = playlist['num_samples']

        # answer will contain all tracks not within the challenge set, but in the original data
        answer['tracks'] = [track for track in answer['tracks'] if track not in playlist['tracks']]

        # keep track of playlist being random or not
        playlist['random'] = random_flag
        answer['random'] = random_flag

        # append the data
        write_data.append(playlist)
        answer_data.append(answer)

    # replace data on the original object with new data
    data['playlists'] = write_data
    data_copy['playlists'] = answer_data

    # dump into the new files
    json.dump(data, fp, indent=4)
    json.dump(data_copy, answers, indent=4)
    fp.close()
    answers.close()


if __name__ == "__main__":
    create_challenge_set()