from fileinput import close

import numpy as np
import scipy as sp
import argparse
from itertools import combinations
from tqdm import tqdm

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
        sparse_matrix.data[:] = 1
    elif matrix_type == 'csc':
        sparse_matrix = sp.sparse.csc_matrix(
            (data[:, 2], (data[:, 0], data[:, 1]))
        )
        sparse_matrix.data[:] = 1
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


def compute_minhash_signatures(matrix, num_hashes, seed=42):
    rng = np.random.default_rng(seed)
    num_users, num_items = matrix.shape
    prime = 4294967311  # A large prime number for hashing
    a = rng.integers(1, prime, size=(num_hashes, 1), dtype=np.int64)
    b = rng.integers(0, prime, size=(num_hashes, 1), dtype=np.int64)
    signatures = np.full((num_hashes, num_users), np.inf)
    for user in tqdm(range(num_users), desc="1/3 Computing MinHash Signatures"):
        start, end = matrix.indptr[user], matrix.indptr[user + 1] #use indptr for speed
        item_indices = matrix.indices[start:end]

        if len(item_indices) == 0:
            continue
        hashes = (a * item_indices + b) % prime
        signatures[:, user] = np.min(hashes, axis=1)
    return signatures.astype(np.int64)


def lsh(signatures, num_bands, rows_per_band):
    num_hashes, num_users = signatures.shape
    assert num_hashes == num_bands * rows_per_band, "num_hashes must equal num_bands * rows_per_band"

    candidate_pairs = set()

    for band in tqdm(range(num_bands), desc="2/3 LSH Bands"):
        start_row = band * rows_per_band
        end_row = start_row + rows_per_band
        band_hashes = {}

        for user in range(num_users):
            band_signature = tuple(signatures[start_row:end_row, user])
            if band_signature not in band_hashes:
                band_hashes[band_signature] = []
            band_hashes[band_signature].append(user)

        for users in band_hashes.values():
            if len(users) > 1:
                candidate_pairs.update(combinations(users, 2))

    return candidate_pairs


def verify_candidates(csr_mat, candidate_pairs, threshold=0.5):
    true_similar_pairs = []

    for u1, u2 in tqdm(candidate_pairs, desc="3/3 Verifying Candidates"):

        items_u1 = csr_mat.indices[csr_mat.indptr[u1]:csr_mat.indptr[u1+1]]
        items_u2 = csr_mat.indices[csr_mat.indptr[u2]:csr_mat.indptr[u2+1]]

        intersection = len(np.intersect1d(items_u1, items_u2, assume_unique=True))
        union = len(items_u1) + len(items_u2) - intersection

        if union > 0:
            jaccard = intersection / union
            if jaccard > threshold:
                true_similar_pairs.append((u1, u2, jaccard))
                write_results(pair=(u1, u2))

    return true_similar_pairs


def write_results(pair):
    with open("results.txt", "a") as f:
        f.write(str(pair[0])+','+str(pair[1])+'\n')
    f.close()


def main():
    # Load the data
    csr_mat = read_data("user_movie_rating.npy", matrix_type="csr")

    # Parameters
    num_hashes = 100
    num_bands = 20
    rows_per_band = num_hashes // num_bands

    # Compute MinHash signatures
    signatures = compute_minhash_signatures(csr_mat, num_hashes)

    # Perform LSH to find candidate pairs
    candidate_pairs = lsh(signatures, num_bands, rows_per_band)

    # Verify candidate pairs using Jaccard similarity
    true_similar_pairs = verify_candidates(csr_mat, candidate_pairs, threshold=0.5)

    print(f"Number of candidate pairs: {len(candidate_pairs)}")
    print(f"Number of true similar pairs: {len(true_similar_pairs)}")
    print("True similar pairs (user1, user2, jaccard_similarity):")
    for pair in true_similar_pairs:
        print(pair)


if __name__ == '__main__':
    main()