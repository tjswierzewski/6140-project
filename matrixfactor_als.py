import numpy as np
import scipy.sparse as ssparse
import create_numpy_from_data as cnfd

'''
Function: matrxifactor_als
Description: Performs matrix factorization using alternating least squares on a given 
dataset in the form of a playlist x song matrix. Formula is as follows:

p_u = ((S.T*S) + lambda*I)^-1 * (S.T * pref_u), where pu is a playlist x feature VECTOR.
s_i = ((P.T*P) + lambda*I)^-1 * (P.T * pref_i), where su is a song x feature VECTOR.

p_u : playlist x feature vector, over all u, this forms P
s_i : song x feature vector, over all i, this forms S
P : playlist x feature matrix, randomly intialized
S : songs x feature matrix, randomly initialized
lambda : Regularization term
I : Identity matrices.
pref_u : a preference row vector for a playlist u and each song
pref_i : a preference column vector for a song i and each playlist
Y : 
Parameters:
    data -- The playlist x song matrix to perform ALS matrix factorization on.
    count -- The number of times to perform the alternating updates to our factored feature matrices.
    features -- The number of latent features to pull out of the dataset.
    lambda_r -- The regularization term used to prevent large values in the factored feature matrices.
Returns:
    Two factored feature matrices in the form playlists x features and songs x features.
'''
def matrixfactor_als(data, count, features, lambda_r):
    print("Starting")
    # Getting size of factored feature matrices.
    p_size, s_size = data.shape
    # Creating factored feature matrices and randomly assigning values according to ALS method.
    p_dense = np.random.normal(size=(p_size, features))
    s_dense = np.random.normal(size=(s_size, features))
    # Sparse feature matrices in CSR format.
    p_sparse = ssparse.csr_matrix(p_dense)
    s_sparse = ssparse.csr_matrix(s_dense)
    # Sparse identity matrices in COOrdinate format.
    fi_sparse = ssparse.eye(features) # Feature identity matrix.
    lamb = np.dot(lambda_r, fi_sparse)
    # Matrix factoring.
    for i in range(0, count, 1):
        print("Calculating iteration", i)
        # Pre-compute sTs and pTp.
        sTs = s_sparse.T.dot(s_sparse)
        pTp = p_sparse.T.dot(p_sparse)
        sTs_lamb = sTs + lamb
        pTp_lamb = pTp + lamb
        for u in range(0, p_size, 1):
            print("Calculating pref u row", u)
            # Computing p_u.
            sTpref_u = s_sparse.T.dot(data[u,:])
            p_sparse[u] = ssparse.linalg.spsolve(sTs_lamb, sTpref_u.T)
        for i in range(0, s_size, 1):
            print("Calculating pref i column", i)
            pTpref_i = p_sparse.T.dot(data[:,i])
            s_sparse[i] = ssparse.linalg.spsolve(pTp_lamb, pTpref_i)
    # Returning.
    return p_sparse, s_sparse

def main():
    df = ssparse.load_npz("UvS_sparse_matrix_D1.npz").toarray()
    p_sparse, s_sparse = matrixfactor_als(df, 10, 10, 0.1)

if (__name__ == "__main__"):
    main()