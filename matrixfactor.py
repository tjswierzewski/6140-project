import numpy as np
import scipy.sparse as ssparse
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from create_numpy_from_data import swap_song_index_to_X, data_to_query_label
from scoring import full_score
'''
DEFINITIONS
'''
TOP_X = 500
MAX_ITER = 500
STATE_RAND = 1
'''
Function: computeFeatureVectors
This function factors a matrix into two feature matrices using non-negative
matrix factorization techniques.
Parameters:
    data | matrix (playlists x songs) -- The data matrix to be factored.
    components | int -- The number of features to pull from the data.
    r | int -- The regularization term.
Returns:
    p_features | matrix (playlists x features)
    s_features | matrix (songs x features)
'''
def computeFeatureVectors(data, components, r):
    factor_model = NMF(n_components=components, init='nndsvd', alpha_W=r,\
                       random_state=STATE_RAND, max_iter=MAX_ITER)
    p_features = factor_model.fit_transform(data)
    s_features = factor_model.components_.transpose()
    return p_features, s_features
'''
Function: predictPlaylistFeatures
This function predicts playlist feature vectors from known playlist 
song data and a known song feature matrix.
Parameters:
    data | matrix (playlists x songs) -- Playlists whose song data is
    partially known and whose feature vectors need to be calculated.
    s_features | matrix (songs x features) -- Songs whose feature
    vectors are known.
Procedure:
    For playlists in the testing set whose feature vectors were not
    initially computed via matrix factorization, but whose subset
    of songs are known, the playlist features can be estimated using
    a standard linear regression model:
    
    Y = X'W, where W needs to be solved for and:
    Y  : (songs x features)
    X' : (songs x playlists)
    W  : (playlists x features)
Returns:
    p_features | matrix (playlists x features) -- Playlists whose
    feature vectors are calculated.
'''
def predictPlaylistFeatures(data, s_features):
    # Obtaining regularization model.
    reg_model = Ridge(fit_intercept=False, positive=True)
    reg_model.fit(data.transpose(), s_features)
    # Obtaining feature vector.
    p_features = reg_model.coef_.transpose()
    return (p_features)
'''
Function: scoring
This function performs scoring operations for the computed
playlist data.
Parameters:
    data | matrix (playlists x songs) -- The computed song data for each
    test query playlist.
    answers | list of lists -- The songs that were actually a part of each
    test query playlist's truncated data.
Returns:
    None.
'''
def scoring(data, answers):
    # Sorting playlists.
    for i in range (data.shape[0]):
        data[i] = data[i].argsort()[::-1]
    # Limiting to top recommendations.
    recommendations = data[:,0:TOP_X]
    # Getting scores.
    zero_count = 0
    loss_count = 0
    avg_clicks = 0
    avg_r = 0
    avg_gain = 0
    size = recommendations.shape[0]
    for i in range (size):
        r, gain, clicks = full_score(answers[i], recommendations[i])
        if clicks == 0:
            zero_count += 1
        elif clicks == 51:
            loss_count += 1
        avg_r += r
        avg_gain += gain
        avg_clicks += clicks
    # Averages.
    avg_r /= size
    avg_gain /= size
    avg_clicks /= size
    # Printing.
    print("Avg R Precision: " + str(avg_r))
    print("Avg Cumulative Gain: " + str(avg_gain))
    print("Avg Clicks: " + str(avg_clicks))
    print("Zero Click Count: " + str(zero_count))
    print("Loss Click Count: " + str(loss_count))
'''
Function: main
This function is the main driver of the program.
Parameters:
    None.
Returns:
    None.
'''
def main():
    print("\nPreparing data...")
    df = ssparse.load_npz("UvS_sparse_matrix_D100.npz")
    # Splitting train and test data.
    train, test = train_test_split(df, test_size = .001, random_state=STATE_RAND)
    # Creating queries and answers.
    test_playlists, test_answers = data_to_query_label(test)
    # Swapping.
    train = swap_song_index_to_X(train)
    test_playlists = swap_song_index_to_X(test_playlists, \
                                          shape=(len(test_playlists), train.shape[1]))
    print("Done!")
    print(train.shape)
    print(test_playlists.shape)
    # Obtaining feature matrices.
    print("\nObtaining feature matrices...")
    p_features, s_features = computeFeatureVectors(train, 70, 0.0001)
    print("Done!")
    # Predicting training playlists.
    print("\nPredicting test playlist features...")
    p_features = predictPlaylistFeatures(test_playlists, s_features)
    print("Done!")
    # Computing dot product to obtain new playlist data.
    print("\nPredicting playlist recommendations...")
    new_playlists = np.dot(p_features, s_features.transpose())
    print("Done!")
    # Scoring.
    print("\nScoring...")
    scoring(new_playlists, test_answers)
    print("Done!")
if (__name__ == "__main__"):
    main()
