from MachineLearning import kneighbors
import numpy as np
import cupy as cp


def distance_matrix(f1):
    shape = f1.shape[0]
    if shape > 5000:
        D = np.zeros((shape, shape), float)
        for i in range(shape):
            for j in range(i + 1, shape):
                dist = np.linalg.norm(f1[i, :] - f1[j, :])
                D[i, j] = dist
                D[j, i] = dist
        return D
    else:
        pairwise_dist = np.sum((f1[:, np.newaxis, :] - f1[np.newaxis, :, :]) ** 2, axis=-1)

        # Return square root of pairwise distances to obtain Euclidean distance
        euclidean_dist = np.sqrt(pairwise_dist)
        return euclidean_dist


def ssm_construction(feature_list, length, k):
    print("\n********** SSM construction start ***********")
    ssm = np.zeros((length, length), dtype='float64')
    normalized_sparse_ssm = np.zeros((length, length), dtype='float64')

    print("\nDistance matrix construction start...")
    dm = distance_matrix(feature_list)
    scaled_distance_mean = -dm.mean() * 0.05
    neighbors = kneighbors(dm, length, k)
    print("Done")
    print("\nSparse ssm and normalized sparse ssm construction start...")
    # construct sparse_ssm
    for i in range(length):
        ith_neigh = neighbors[i]
        ssm_elements = np.exp(dm[i][ith_neigh] / scaled_distance_mean)
        ssm[i][ith_neigh] = ssm_elements
        ssm_sum = 2.0 * (np.sum(ssm_elements) - 1.)
        if ssm_sum != 0:
            normalized_sparse_ssm[i][ith_neigh] = ssm_elements / ssm_sum
        normalized_sparse_ssm[i][i] = 0.5
    print("Done\n")

    trium = np.triu(normalized_sparse_ssm, k=1)
    normalized_sparse_ssm = trium + trium.T + np.diag(normalized_sparse_ssm.diagonal())

    return ssm, normalized_sparse_ssm


def neighbor_matrix(neighbors, length, n_neigh):
    i_neigh = np.zeros((length, n_neigh ** 2), dtype=float)
    j_neigh = np.zeros((length, n_neigh ** 2), dtype=float)
    for i in range(length):
        duplicate_neighbors = sorted(list(neighbors[i])) * n_neigh
        j_neigh[i] = duplicate_neighbors
        i_neigh[i] = sorted(duplicate_neighbors)
    return i_neigh, j_neigh


def process_diffusion(fused_ssm, dfssm, i_neigh, j_neigh, neighbors, k2, t2, rweight, length):
    # construct enhanced ssm leveraging "fused ssm (fused_ssm)" and "denoised fused ssm (dfssm)"
    print("\nSSM enhancement start...")
    enhanced_ssm = cp.zeros((length, length), dtype='float64')

    neighbors = neighbors.astype(dtype='int32')
    neighbors = np.sort(neighbors, axis=1)
    ci_neigh = cp.asarray(i_neigh, dtype='int32')
    cj_neigh = cp.asarray(j_neigh, dtype='int32')

    essm = fused_ssm.copy()

    for _t2 in range(t2):
        for i in range(length):
            ith_neigh = neighbors[i]
            Qi = dfssm[i, ith_neigh]
            vstacks = cp.random.random((k2,), dtype=float)
            c_ith_neigh = ci_neigh[i]
            for j in range(length):
                A = essm[c_ith_neigh, cj_neigh[j]].reshape((k2, k2))
                Qj = dfssm[neighbors[j], j]
                vstacks = cp.vstack((vstacks, cp.matmul(A, Qj)))
            vstacks = vstacks[1:].T
            ith_essm = cp.matmul(Qi, vstacks)
            enhanced_ssm[i, :] = ith_essm
        essm = (rweight * enhanced_ssm + (1.0 - rweight) * dfssm).copy()

    print("Done")
    print("*********************************************")
    essm = cp.asnumpy(essm)
    return essm


def ssm_enhancement(fused_ssm, k2, t2, rweight):
    print("\n********** SSM enhancement start ***********")
    length = fused_ssm.shape[0]

    neighbors = kneighbors(fused_ssm, length, k2)

    print("\nLocalized fused ssm construction start...")
    # construct the localized fused ssm by using KNN
    lfssm = cp.zeros((length, length), dtype='float64')  # nfssm means knn based "normalized fused ssm"
    fused_ssm = cp.asarray(fused_ssm)
    for i in range(length):
        ith_neighs = neighbors[i]
        lfssm[i][ith_neighs] = fused_ssm[i][ith_neighs] / cp.sum(fused_ssm[i][ith_neighs])

    print("Done\n")
    print("Denoised fused ssm construction start...")
    # construct the denoised fused ssm exploiting nfssm
    dfssm = cp.zeros((length, length), dtype='float64')  # dfssm means denoised fused ssm

    sum_lfssm = cp.sum(lfssm, axis=0)
    # upper trianluar matrix construction
    for i in range(length):
        for j in range(i, length):
            dfssm[i][j] = cp.sum(lfssm[i, :] * lfssm[j, :] / sum_lfssm)

    # reconstruct above matrix to symmetric matrix
    dfssm += dfssm.T - cp.diag(dfssm.diagonal())

    print("Done\n")

    print("\n Generate table for neighbors ...")
    i_neigh, j_neigh = neighbor_matrix(neighbors, length, k2)
    print("Done\n")

    essm = process_diffusion(fused_ssm, dfssm, i_neigh, j_neigh, neighbors, k2, t2, rweight, length)
    return essm