import numpy as np
import scipy as sp

def read_data(file_path):
    #load data into np.array
    data = np.load(file_path)
    #make a coo sparse matrix with the ratings as values, this one maintains the value of the ratings
    sparse_matrix = sp.sparse.coo_matrix(
        (data[:, 2], (data[:, 0], data[:, 1]))
    )
    # if necessary, we can also use the following code to change all ratings to 1 instead:
    #lil = sparse_matrix.tolil()
    #lil[lil != 0] = 1
    #sparse_matrix = lil.tocoo()
    return sparse_matrix



if __name__ == '__main__':
    # read the data
    data = read_data("user_movie_rating.npy")
    print(data)