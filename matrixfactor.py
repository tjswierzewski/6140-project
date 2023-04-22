import numpy as np
import scipy.sparse as ssparse
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from create_numpy_from_data import swap_song_index_to_X, data_to_query_label, Song, SongList
from scoring import full_score
'''
DEFINITIONS
'''
TOP_X = 500
MAX_ITER = 500
STATE_RAND = 1
FEATURES = 10
QUERY_LENGTH = 10
KEY_LENGTH = 10
TEST_PLAYLIST_COUNT = 1000
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
    print(data.shape)
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
    data | matrix (playlist x songs) -- Playlist whose song data is
    partially known and whose feature vectors need to be calculated.
    s_features | matrix (songs x features) -- Songs whose feature
    vectors are known.
Procedure:
    For playlists in the testing set whose feature vectors were not
    initially computed via matrix factorization, but whose subset
    of songs are known, the playlist features can be estimated using
    a standard linear regression model:
    
    Y = X'W, where W needs to be solved for and:
    Y  : (songs x playlists)
    X' : (features x songs)
    W  : (features x playlists)
Returns:
    A playlists whose feature vector is calculated.
'''
def predictPlaylistFeatures(playlist, s_features):
    # Densifying.
    playlist = playlist.toarray()
    # Obtaining regularization model.
    reg_model = Ridge(random_state=STATE_RAND)
    # Fitting data.
    reg_model.fit(s_features, playlist.transpose())
    # Obtaining playlist feature vector.
    p_features = reg_model.coef_
    return p_features
'''
Function: removeDuplicates
This function removes duplicates from a recommendations list 
with respect to an existing playlist for which the recommendations
are being made.
Parameters:
    recommendations | list -- A list of song recommendations for a 
    given playlist.
    existing_playlist | list -- The given playlist for which the
    recommendations were made.
Returns:
    The updated list of recommendations.
'''
def removeDuplicates(recommendations, existing_playlist):
    i = 0
    while i < (len(recommendations)):
        if recommendations[i] in existing_playlist:
            recommendations = np.delete(recommendations, i)
        else:
            i += 1
    return recommendations
'''
Function: getRecommendations
This function gets the recommendations for each playlist in a list
of playlists. Recommendations are truncated to a specified size.
Parameters:
    playlists | list of matrices -- The playlists to generate 
    recommendations for.
    s_features | matrix (songs x features) -- Songs whose feature
    vectors are known.
    test_queries | list of lists -- The same playlists for which
    recommendations are being generated, in a different format for
    the purpose of duplicat removal.
Returns:
    The list of all recommendations for each playlist.
'''
def getRecommendations(playlists, s_features, test_queries):
    # Storage for recommendations.
    recommendation_list = np.zeros((playlists.shape[0], TOP_X))
    # For each playlist.
    for i in range(0, playlists.shape[0]):
        # Calculate its feature vector.
        p_features = predictPlaylistFeatures(playlists[i], s_features)[0]
        # Generate a new playlist, sort by most likely song, and gather top + 10 recommendations.
        recommendations = np.dot(p_features, s_features.transpose()).argsort()[::-1][:TOP_X + QUERY_LENGTH]
        # Remove any songs in original playlist.
        recommendations = removeDuplicates(recommendations, test_queries[i])
        # Truncate to 500 recommendations and store.
        recommendation_list[i] = recommendations[:TOP_X]
    return recommendation_list
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
def scoring(recommendations, answers):
    # Stats.
    zero_count = 0
    loss_count = 0
    avg_clicks = 0
    avg_r = 0
    avg_gain = 0
    # Amount of playlists.
    size = recommendations.shape[0]
    # Scoring playlists.
    for i in range(size):
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
    print("load")
    # Loading data.
    df = ssparse.load_npz("UvS_sparse_matrix_D10.npz")
    # Splitting train and test data.
    train, test = train_test_split(df, test_size=.1, random_state=STATE_RAND)
    # Creating queries and answers from test data.
    test_queries, test_answers = data_to_query_label(test, query_length=QUERY_LENGTH, \
                                                     min_key_length=KEY_LENGTH, \
                                                     max_return=TEST_PLAYLIST_COUNT)
    print("swap")
    # Swapping.
    train = swap_song_index_to_X(train, shape=None, reducer='binary')
    test = swap_song_index_to_X(test_queries, shape=(len(test_queries), train.shape[1]), reducer='binary')
    print("MF")
    # Obtaining feature matrices.
    p_features, s_features = computeFeatureVectors(train, FEATURES, 0.0001)
    #np.savetxt("s_D1000_" + str(FEATURES) + ".csv", s_features, delimiter=",")
    # Predicting training playlists.
    #s_features = np.load("s_D1000_70.npz")
    #s_features = s_features[s_features.files[0]]
    # Recommending songs for a maximum of 1000 test playlists..
    print("reccs")
    print(test.shape)
    recommendations = getRecommendations(test, s_features, test_queries)
    # Scoring.
    scoring(recommendations, test_answers)
if __name__ == "__main__":
    main()