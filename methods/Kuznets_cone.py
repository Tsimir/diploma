import numpy as np
from scipy.optimize import nnls

from utility import saver
from utility.generator import (matrix_to_ranks, matrix_to_ordering, compute_my_kendall,
                               ranks_to_matrix, get_best_tolerance)


def find_weights_cone_direct(expert_matrices: list[np.ndarray],
                             target_matrix: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Прямой конусный метод.
    Модель: f = sum(Z_j * lambda_j), lambda_j >= 0
    """
    m = target_matrix.shape[0]
    n = len(expert_matrices)

    target_vec = target_matrix @ np.ones(m)
    lambdas = [np.ones(m) / m for _ in range(n)]

    for _ in range(max_iter):
        for j in range(n):
            residual = target_vec.copy()
            for k in range(n):
                if k != j:
                    residual -= expert_matrices[k] @ lambdas[k]
            lambdas[j], _ = nnls(expert_matrices[j], residual)

    result_ranks = np.sum([expert_matrices[j] @ lambdas[j] for j in range(n)], axis=0)

    return result_ranks


def find_weights_cone_regularized(expert_matrices: list[np.ndarray],
                                  target_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Регуляризованный конусный метод.
    """
    m = target_matrix.shape[0]
    n = len(expert_matrices)

    target_vec = target_matrix.flatten()
    expert_vecs = np.column_stack([expert.flatten() for expert in expert_matrices])

    w, _ = nnls(expert_vecs, target_vec)
    weights = np.round(w, 3)

    preresult_matrix = np.sum([weights[j] * expert_matrices[j] for j in range(n)], axis=0)

    target_vec2 = target_matrix @ np.ones(m)
    lambda_raw, _ = nnls(preresult_matrix, target_vec2)

    lambda_norm = lambda_raw / lambda_raw.sum()

    result_ranks = preresult_matrix @ lambda_norm

    return weights, result_ranks


def find_weights(target_matrix: np.ndarray, expert_matrices: list[np.ndarray],
                 method: str = 'cone_direct', tolerance: float = 0.01) -> dict:
    """
    Конусные методы агрегирования экспертных предпочтений.

    Параметры:
        target_matrix — целевая матрица предпочтений
        expert_matrices — список экспертных матриц
        method — 'cone_direct' (прямой) или 'cone_reg' (регуляризованный)
        tolerance — параметр толерантности для группировки
    """
    if method == 'cone_direct':
        result_ranks = find_weights_cone_direct(expert_matrices, target_matrix)

        best_tol = get_best_tolerance(result_ranks, target_matrix)

        result_matrix = ranks_to_matrix(result_ranks, tolerance)
        ideal_result_matrix = ranks_to_matrix(result_ranks, best_tol)

        hon_ordering = matrix_to_ordering(result_matrix, tolerance)
        ideal_ordering = matrix_to_ordering(ideal_result_matrix, best_tol)

        hon_ranking = matrix_to_ranks(result_matrix, tolerance)
        ideal_ranking = matrix_to_ranks(ideal_result_matrix, best_tol)

        error = np.linalg.norm(target_matrix - result_matrix, 'fro')

        hbin_result_matrix = ranks_to_matrix(hon_ranking)
        ibin_result_matrix = ranks_to_matrix(ideal_ranking)

        return {
            'row_result_matrix': result_matrix,
            'hon_ranking': hon_ranking,
            'ideal_ranking': ideal_ranking,
            'hon_ordering': hon_ordering,
            'ideal_ordering': ideal_ordering,
            'error': error,
            'hon_bin_result_matrix': hbin_result_matrix,
            'ideal_bin_result_matrix': ibin_result_matrix
        }
    else:
        weights, result_ranks = find_weights_cone_regularized(expert_matrices, target_matrix)

        best_tol = get_best_tolerance(result_ranks, target_matrix)

        result_matrix = ranks_to_matrix(result_ranks, tolerance)
        ideal_result_matrix = ranks_to_matrix(result_ranks, best_tol)

        hon_ordering = matrix_to_ordering(result_matrix, tolerance)
        ideal_ordering = matrix_to_ordering(ideal_result_matrix, best_tol)

        hon_ranking = matrix_to_ranks(result_matrix, tolerance)
        ideal_ranking = matrix_to_ranks(ideal_result_matrix, best_tol)

        error = np.linalg.norm(target_matrix - result_matrix, 'fro')

        hbin_result_matrix = ranks_to_matrix(hon_ranking)
        ibin_result_matrix = ranks_to_matrix(ideal_ranking)

        return {
            'row_result_matrix': result_matrix,
            'hon_ranking': hon_ranking,
            'ideal_ranking': ideal_ranking,
            'hon_ordering': hon_ordering,
            'ideal_ordering': ideal_ordering,
            'error': error,
            'hon_bin_result_matrix': hbin_result_matrix,
            'ideal_bin_result_matrix': ibin_result_matrix
        }

if __name__ == "__main__":
    all_matrices = saver.read_matrices_from_csv("../test_matrices/noise_matrices.csv")

    n = len(all_matrices) - 1
    m = len(all_matrices[0])
    base_tolerance = 0.6

    expert_matrices = all_matrices[:n]
    target_matrix = all_matrices[n]

    print(f"Данные: {n} экспертов, {m} объектов\n")

    print("\nИстинное ранжирование объектов (от лучшего к худшему):")
    ordering = matrix_to_ordering(target_matrix)

    position = 1
    for cluster in ordering:
        if len(cluster) == 1:
            print(f"  {position}. Объект {cluster[0]}")
        else:
            print(f"  {position}. Группа {cluster}")
        position += 1

    print(f"\nРанги: {matrix_to_ranks(target_matrix, base_tolerance)} \n")

    if n >= m:
        print(f"\nРанговый метод не имеет предсказательной силы из-за недоопределенности!\n")
    if n >= m * m:
        print(f"Матричный метод также не имеет предсказательной силы!\n")

    method_names = {
        'cone_direct': 'Конусный (прямой)',
        'cone_reg': 'Конусный (регуляризованный)'
    }

    for method in ['cone_direct', 'cone_reg']:
        print(f"{'=' * 40}")
        print(f"Метод: {method_names[method]}")
        print(f"{'=' * 40}")

        result = find_weights(target_matrix, expert_matrices, method=method, tolerance=base_tolerance)

        print("\nВеса экспертов:")
        if method == 'cone_direct':
            print("  (Прямой метод не вычисляет веса экспертов)")
        else:
            for i, weight in enumerate(result['weights']):
                print(f"  Эксперт {i + 1:2}: {weight:.3f}")

        print("\nРанжирование объектов (от лучшего к худшему):")
        ordering = result['ideal_ordering']

        position = 1
        for cluster in ordering:
            if len(cluster) == 1:
                print(f"  {position}. Объект {cluster[0]}")
            else:
                print(f"  {position}. Группа {cluster}")
            position += 1

        print(f"\nРанги: {result['ideal_ranking']}")
        print(f"tau: {compute_my_kendall(result['ideal_bin_result_matrix'], target_matrix):.3f}")
        print()