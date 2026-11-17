import numpy as np
import scipy as sp
import argparse

def read_data(file_path, matrix_type='coo'):
    #load data into np.array
    data = np.load(file_path)
    #make a coo sparse matrix with the ratings as values, this one maintains the value of the ratings
    if matrix_type == 'coo':
        sparse_matrix = sp.sparse.coo_matrix(
            (data[:, 2], (data[:, 0], data[:, 1]))
        )
    elif matrix_type == 'csr':
        sparse_matrix = sp.sparse.csr_matrix(
            (data[:, 2], (data[:, 0], data[:, 1]))
        )
    elif matrix_type == 'csc':
        sparse_matrix = sp.sparse.csc_matrix(
            (data[:, 2], (data[:, 0], data[:, 1]))
        )
    elif matrix_type == 'lil':
        sparse_matrix = sp.sparse.lil_matrix(
            (data[:, 2], (data[:, 0], data[:, 1]))
        )
    else:
        raise ValueError("Invalid matrix type. Choose from 'coo', 'csr', 'csc', or 'lil'.")

    # if necessary, we can also use the following code to change all ratings to 1 instead:
    #lil = sparse_matrix.tolil()
    #lil[lil != 0] = 1
    #sparse_matrix = lil.tocoo()
    return sparse_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read user-movie rating data and convert to sparse matrix.')
    parser.add_argument("random_seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    # read the data
    data = read_data("user_movie_rating.npy")
    np.random.seed(args.random_seed)
    print(data)