# TODO:
* run pipeline again
* write paper

# Overview

This repository contains the original recommender pipeline from our <a href="https://dl-acm-org.ezp3.lib.umn.edu/doi/pdf/10.1145/3267471.3267488">paper</a> in addition to our enhancements. The original repository can be found 
<a href="https://github.com/VasiliyRubtsov/recsys2018">here</a>


# instructions

## data preparation
* download any number of playlist slices from <a href="https://www.kaggle.com/datasets/himanshuwagh/spotify-million">the million playlist dataset</a>, and download one extra slice to be used as "challenge set"
* the testing slices are to be kept in a _/data_ directory, while the challenge set slice should be renamed to _challenge_set_data.json_ and kept in the top level directory
* Once this is complete, run the _create_challenge_set.py_ file in order to split the challenge set data as required by the paper
* for the item-item recommender, additional datasets that encapsulate song features are required, they can be downloaded <a href="https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm?select=User+Listening+History.csv">here</a> and <a href="https://www.kaggle.com/datasets/tomigelo/spotify-audio-features">here</a>. They must be kept with their original names, and placed in the _/data_ directory.

## original pipeline
the original paper includes 8 python notebook files that are to be run in sequence. The order is as follows:

1) json_to_dataframe.ipynb
2) validation_strategy.ipynb
3) lightfm.ipynb
4) lightfm_text.ipynb
5) candidate_selection.ipynb
6) lightfm_features.ipynb
7) co_occurence_features.ipynb
8) xgboost.ipynb

The execution time of this pipeline is heavily dependent on the number of playlist slices downloaded, but in this study it was ~1 hour.
Once this is completed, a "submission.hdf" file will be present with the predicted tracks for each playlist.

## our additions
In addition to the 8 files above, we have 3 more python notebook files that are to be run
1) baseline.ipynb
2) item_item.ipynb
3) interpret_results.ipynb

The results of this are displayed at the end of the interpret_results.ipynb file, where the original, baseline, and item item predictions are measured with a handful of different metrics