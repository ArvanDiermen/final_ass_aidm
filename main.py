import numpy as np
import scipy as sp
import argparse
import time

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

from itertools import combinations
import numpy as np
from tqdm import tqdm
def compute_minhash_signatures(matrix, num_hashes, rng):
    # matrix is a CSR matrix of shape (num_users, num_movies)
    num_users, num_movies = matrix.shape
    indptr, indices = matrix.indptr, matrix.indices
    nonempty = np.diff(indptr) > 0
    starts = indptr[:-1][nonempty]
    signatures = np.full((num_hashes, num_users), -1, dtype=np.int32)
    for h in range(num_hashes):
        # a random order of all movies (instead of a hash function, Hint 2)
        perm = rng.permutation(num_movies).astype(np.int32)
        values = perm[indices]
        signatures[h, nonempty] = np.minimum.reduceat(values, starts)
    return signatures

def lsh(signatures, num_bands, rows_per_band, rng, max_bucket_size=50):
    # signatures has shape (num_hashes, num_users): one column per user
    num_hashes, num_users = signatures.shape
    assert num_hashes >= num_bands * rows_per_band, "not enough rows in signatures"

    # random multipliers, used to squeeze the r values of a band into ONE number
    multipliers = rng.integers(1, 2**61, size=rows_per_band, dtype=np.int64)

    # users with an empty signature (-1) are ignored (this is only user 0, which doesn't exist)
    valid_users = np.flatnonzero(signatures[0] >= 0)

    candidate_pairs = set()
    skipped_buckets = 0

    for band in range(num_bands):
        start_row = band * rows_per_band
        end_row = start_row + rows_per_band

        # take the rows of this band: shape (rows_per_band, num_valid_users)
        block = signatures[start_row:end_row, valid_users].astype(np.int64)

        # combine the r rows of every user into a single number (the "bucket key"),
        # users with identical band content get the same key (overflow just wraps around, which is fine)
        keys = (block * multipliers[:, None]).sum(axis=0)
        
        # sort users by key, so users with the same key (= same bucket) are next to each other
        # (stable sort keeps the users in increasing order inside each bucket, so u1 < u2 always)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_users = valid_users[order]

        # find where the key changes: every change is the start of a new bucket
        change_points = np.flatnonzero(np.diff(sorted_keys)) + 1
        bucket_starts = np.concatenate(([0], change_points))
        bucket_ends = np.concatenate((change_points, [len(sorted_keys)]))
        bucket_sizes = bucket_ends - bucket_starts

        # only buckets with at least 2 users can contain candidate pairs
        for s, e, size in zip(bucket_starts[bucket_sizes >= 2],
                              bucket_ends[bucket_sizes >= 2],
                              bucket_sizes[bucket_sizes >= 2]):
            # huge buckets would create millions of pairs, so we skip them (see Hint 4)
            if size > max_bucket_size:
                skipped_buckets += 1
                continue
            # every two users in the same bucket become a candidate pair
            candidate_pairs.update(combinations(sorted_users[s:e].tolist(), 2))

    print(f"Skipped {skipped_buckets} buckets bigger than {max_bucket_size}")
    return candidate_pairs   
def verify_candidates(csr_mat, candidate_pairs, out_path, threshold=0.5):
    found = 0
    for u1, u2 in candidate_pairs:
        items_u1 = csr_mat.indices[csr_mat.indptr[u1]:csr_mat.indptr[u1 + 1]]
        items_u2 = csr_mat.indices[csr_mat.indptr[u2]:csr_mat.indptr[u2 + 1]]

        intersection = len(np.intersect1d(items_u1, items_u2, assume_unique=True))
        union = len(items_u1) + len(items_u2) - intersection

        if union > 0 and intersection / union > threshold:   
            a, b = min(u1, u2), max(u1, u2)               
            with open(out_path, "a") as f:                  
                f.write(f"{a},{b}\n")
            found += 1
    return found

def main():
    #read the random seed from the command line (required by the assignment)
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, help="random seed")
    # optional arguments, so we can try different parameters without editing the code
    parser.add_argument("--bands", type=int, default=25, help="number of bands (b)")
    parser.add_argument("--rows", type=int, default=6, help="rows per band (r)")
    parser.add_argument("--max_bucket", type=int, default=200, help="skip buckets bigger than this")
    args = parser.parse_args()

    # one random number generator for the whole program, so the same seed gives the same result
    rng = np.random.default_rng(args.seed)

    # prepare the output file
    out_path = "result.txt"
    open(out_path, "w").close()          # empty the file, in case an old run left results in it

    #load the data as a sparse User x Movie matrix (CSR format)
    t0 = time.time()
    csr_mat = read_data("user_movie_rating.npy", matrix_type="csr")
    print(f"Data loaded: {time.time() - t0:.0f}s")

    # parameters (temporary values, we tune them later) 
    num_bands = args.bands
    rows_per_band = args.rows
    num_hashes = num_bands * rows_per_band 

    #compute the minhash signatures (one column per user)
    signatures = compute_minhash_signatures(csr_mat, num_hashes, rng)
    print(f"Signatures done: {time.time() - t0:.0f}s")

    # LSH banding, users that share a bucket in some band become candidate pairs
    candidate_pairs = lsh(signatures, num_bands, rows_per_band, rng, max_bucket_size=args.max_bucket)
    print(f"Candidates: {len(candidate_pairs)}  ({time.time() - t0:.0f}s)")

    #check the candidates with the real Jaccard similarity,
    # and write the pairs with similarity > 0.5 to result.txt as "u1,u2"
    found = verify_candidates(csr_mat, candidate_pairs, out_path)
    print(f"Similar pairs found: {found}  (total {time.time() - t0:.0f}s)")
if __name__ == "__main__":
    main()
