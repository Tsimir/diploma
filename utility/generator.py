from typing import List

import numpy as np
from scipy.stats import kendalltau, spearmanr


def PrefMaGen_direct(m: int, p_dominate: float = 0.3) -> np.ndarray:

    matrix = np.eye(m, dtype=np.int8)

    for i in range(m):
        for j in range(i + 1, m):
            rand = np.random.rand()
            if rand < p_dominate:
                matrix[i, j] = 1
                matrix[j, i] = 0
            elif rand < 2 * p_dominate:
                matrix[j, i] = 1
                matrix[i, j] = 0
            else:
                matrix[i, j] = 0
                matrix[j, i] = 0

    for i in range(m):
        for k in range(m):
            if matrix[k, i] == 1:
                row_i = matrix[i]
                row_k = matrix[k]
                for j in range(m):
                    if row_k[j] == 1 and row_i[j] == 0:
                        matrix[k, j] = 1
                        matrix[j, k] = 0

    for i in range(m):
        for j in range(i + 1, m):
            if matrix[i, j] == 1 and matrix[j, i] == 1:
                if np.random.rand() < 0.5:
                    matrix[j, i] = 0
                else:
                    matrix[i, j] = 0

    return matrix

def PrefMaGen_level(m: int) -> np.ndarray:

    ranks = np.random.permutation(m)
    max_levels = np.random.randint(1, m + 1)
    ranking_levels = [[] for _ in range(max_levels)]

    for obj in ranks:
        level = np.random.randint(0, max_levels)
        ranking_levels[level].append(obj)

    ordering = []
    for level in ranking_levels:
        if level:
            ordering.append([int(x + 1) for x in level])

    return ordering_to_matrix(ordering,m)

def PrefMaGen_linear(m: int) -> np.ndarray:

    permutation = np.random.permutation(m) + 1
    ordering = [[int(x)] for x in permutation]

    return ordering_to_matrix(ordering, m)

def PrefMaGen_noise(Z_base: np.ndarray, noise_level: float = 0.1) -> np.ndarray:

    m = Z_base.shape[0]

    Z_noisy = Z_base.copy().astype(np.int8)

    mask = (np.random.rand(m, m) < noise_level / 2) & (~np.eye(m, dtype=bool))
    Z_noisy[mask] = 1 - Z_noisy[mask]
    Z_noisy[Z_noisy == 1] = 1
    Z_noisy[Z_noisy == 0] = 0

    for i in range(m):
        for j in range(i + 1, m):
            if Z_noisy[i, j] == 1:
                Z_noisy[j, i] = 0
            elif Z_noisy[j, i] == 1:
                Z_noisy[i, j] = 0
            else:
                Z_noisy[i, j] = 0
                Z_noisy[j, i] = 0
    return Z_noisy

def matrix_to_ranks(Z: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """
    Преобразует матрицу  в вектор рангов.
    """
    m = Z.shape[0]
    scores = Z.sum(axis=1)

    sorted_indices = np.argsort(scores)

    ranks = np.zeros(m, dtype=float)
    i = 0
    while i < m:
        group_start = i
        current_score = scores[sorted_indices[i]]
        j = i + 1
        while j < m and abs(scores[sorted_indices[j]] - current_score) < tolerance:
            j += 1
        group_end = j
        group_size = group_end - group_start - 1

        avg_rank = 1 + group_start + group_size / 2.0

        for k in range(group_start, group_end):
            obj_index = sorted_indices[k]
            ranks[obj_index] = avg_rank

        i = group_end

    return ranks

def matrix_to_ordering(Z: np.ndarray, tolerance: float = 0.6) -> List[List[int]]:
    """
    Преобразует матрицу в упорядочение.
    """
    m = Z.shape[0]
    scores = Z.sum(axis=1)

    sorted_indices = np.argsort(scores)[::-1]

    clusters = []
    i = 0
    while i < m:
        current_cluster = [int(sorted_indices[i] + 1)]
        current_score = scores[sorted_indices[i]]

        j = i + 1
        while j < m and abs(scores[sorted_indices[j]] - current_score) < tolerance:
            current_cluster.append(int(sorted_indices[j] + 1))
            j += 1

        clusters.append(current_cluster)
        i = j

    return clusters

def ranks_to_matrix(ranks: np.ndarray, tolerance: float = 0.6) -> np.ndarray:

    m = len(ranks)
    Z = np.eye(m, dtype=np.int8)

    for i in range(m):
        for j in range(i+1, m):
            if abs(ranks[i] - ranks[j]) < tolerance:
                Z[i, j] = 0
                Z[j, i] = 0
            elif (ranks[i] - ranks[j]) >= tolerance:
                Z[i, j] = 1
                Z[j, i] = 0
            elif (ranks[i] - ranks[j]) <= -tolerance:
                Z[i, j] = 0
                Z[j, i] = 1

    return Z

def ordering_to_matrix(ordering: List[List[int]], m: int) -> np.ndarray:

    Z = np.eye(m, dtype=np.int8)

    obj_to_cluster = {}
    for cluster_idx, cluster in enumerate(ordering):
        for obj in cluster:
            obj_to_cluster[obj] = cluster_idx

    for i in range(1, m+1):
        for j in range(1, m+1):
            if i == j:
                continue
            if i in obj_to_cluster and j in obj_to_cluster:
                ci = obj_to_cluster[i]
                cj = obj_to_cluster[j]
                if ci < cj:
                    Z[i - 1, j - 1] = 1
                elif ci > cj:
                    Z[j - 1, i - 1] = 1

    return Z

def compute_my_kendall(Z_1: np.ndarray, Z_2: np.ndarray) -> float:

    m = Z_1.shape[0]

    inversions = 0
    for i in range(m):
        for j in range(i+1, m):
            if Z_1[i,j] != Z_2[i,j]:
                inversions += 1


    tau = 1 - 4 * inversions / (m * (m-1))
    return tau

def compute_my_spearman(ranks1: np.ndarray, ranks2: np.ndarray) -> float:

    m = len(ranks1)

    d_squared_sum = np.sum((ranks1 - ranks2) ** 2)

    rho = 1 - (6 * d_squared_sum) / (m * (m ** 2 - 1))
    return rho

def get_best_tolerance(Z_aprox: np.ndarray, Z_targ: np.ndarray, step: float = 0.01) -> float:

    if Z_aprox.ndim > 1:
        aprox_ranks = matrix_to_ranks(Z_aprox)
    else:
        aprox_ranks = Z_aprox

    best_tau = compute_my_kendall(ranks_to_matrix(aprox_ranks),Z_targ)
    best_tol = 0.6

    tol = 0.0
    while tol < 1.0:
        bin_matrix = ranks_to_matrix(aprox_ranks, tolerance=tol)
        tau = compute_my_kendall(bin_matrix, Z_targ)

        if tau > best_tau:
            best_tau = tau
            best_tol = tol

        tol += step

    return best_tol

if __name__ == "__main__":

    m = 100
    wanted_tau = 0.8
    tol = 0.05

    Z_linear = PrefMaGen_linear(m)
    Z_noisy = PrefMaGen_noise(Z_linear, noise_level=(1-wanted_tau))
    print("Исходная матрица:")
    print(Z_linear)
    print("\nЗашумленная матрица:")
    print(Z_noisy)
    m1 = matrix_to_ranks(Z_linear)
    m2 = matrix_to_ranks(Z_noisy)
    tau, _ = kendalltau(m1, m2)
    rho, _ = spearmanr(m1, m2)
    print("\n wanted tau = ", wanted_tau)
    print("\n bib tau= ", tau, " bib rho= ", rho )
    print("\n my tau= ", compute_my_kendall(Z_linear, Z_noisy) , " my rho= ", compute_my_spearman(m1,m2))

    print("\nУпорядочения (исх + итог): \n", matrix_to_ordering(Z_linear), "\n",matrix_to_ordering(Z_noisy), "\n")