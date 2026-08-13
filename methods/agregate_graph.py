import numpy as np

from utility.generator import (
    matrix_to_ranks, matrix_to_ordering, compute_my_kendall,
    ranks_to_matrix
)


def majority_graph_method(expert_matrices: list[np.ndarray]) -> np.ndarray:
    m = expert_matrices[0].shape[0]

    # Строим матрицу весов (разность голосов)
    weight_matrix = np.zeros((m, m))

    for Z in expert_matrices:
        weight_matrix += Z

    for i in range(m):
        for j in range(m):
            if i != j:
                weight_matrix[i, j] = weight_matrix[i, j] - weight_matrix[j, i]

    np.fill_diagonal(weight_matrix, 0)

    # ориентированный граф: дуга i->j если weight_matrix[i,j] > 0
    smej_mtrx = [[] for _ in range(m)]
    indegree = [0] * m

    for i in range(m):
        for j in range(m):
            if i != j and weight_matrix[i, j] > 0:
                smej_mtrx[i].append(j)
                indegree[j] += 1

    # scores = сумма положительных весов
    scor = np.sum(np.maximum(weight_matrix, 0), axis=1)

    # топологическая сортировка с приоритетом
    zero_indeg = [i for i in range(m) if indegree[i] == 0]
    order = []

    while zero_indeg:
        zero_indeg.sort(key=lambda x: scor[x], reverse=True)
        v = zero_indeg.pop(0)
        order.append(v)

        for to in smej_mtrx[v]:
            indegree[to] -= 1
            if indegree[to] == 0:
                zero_indeg.append(to)

    remaining = [i for i in range(m) if i not in order]
    remaining.sort(key=lambda x: scor[x], reverse=True)
    order.extend(remaining)

    result_ranks = np.zeros(m)
    for pos, obj_idx in enumerate(order):
        result_ranks[obj_idx] = m - pos

    return result_ranks


def find_weights(expert_matrices: list[np.ndarray], target_matrix: np.ndarray = None, tolerance: float = 0.0) -> dict:
    """
    Агрегирует экспертные предпочтения в единое отношение (графовые методы).

    Параметры:
        expert_matrices: список матриц экспертов
        target_matrix: целевая матрица (опционально, для вычисления ошибки)
        tolerance: порог для кластеризации

    Возвращает:
        словарь с результатами
    """

    result_ranks = majority_graph_method(expert_matrices)

    m = expert_matrices[0].shape[0]
    result_matrix = ranks_to_matrix(result_ranks, tolerance)

    hon_ordering = matrix_to_ordering(result_matrix, tolerance)
    hon_ranking = matrix_to_ranks(result_matrix, tolerance)


    error = 0.0
    if target_matrix is not None:
        error = np.linalg.norm(target_matrix - result_matrix, 'fro')

    hbin_result_matrix = ranks_to_matrix(hon_ranking, tolerance)

    return {
        'row_result_matrix': result_matrix,
        'hon_ranking': hon_ranking,
        'hon_ordering': hon_ordering,
        'ideal_ranking': hon_ranking,
        'ideal_bin_result_matrix': hbin_result_matrix,
        'hon_bin_result_matrix': hbin_result_matrix,
        'error': error,
    }


if __name__ == "__main__":
    from utility import saver

    # Загрузка данных
    all_matrices = saver.read_matrices_from_csv("../test_matrices/noise_matrices.csv")

    n = len(all_matrices) - 1
    m = len(all_matrices[0])
    base_tolerance = 0.6

    expert_matrices = all_matrices[:n]
    target_matrix = all_matrices[n]

    print(f"Данные: {n} экспертов, {m} объектов\n")

    print("\nИстинное ранжирование объектов (от лучшего к худшему):")
    ordering = matrix_to_ordering(target_matrix, base_tolerance)

    position = 1
    for cluster in ordering:
        if len(cluster) == 1:
            print(f"  {position}. Объект {cluster[0]}")
        else:
            print(f"  {position}. Группа {cluster}")
        position += 1

    print(f"\nРанги: {matrix_to_ranks(target_matrix, base_tolerance)}\n")

    print(f"{'=' * 40}")
    print(f"Метод: {'Граф большинства'}")
    print(f"{'=' * 40}")

    result = find_weights(expert_matrices, target_matrix, base_tolerance)

    print("\nРанжирование объектов (от лучшего к худшему):")
    ordering = result['hon_ordering']

    position = 1
    for cluster in ordering:
        if len(cluster) == 1:
            print(f"  {position}. Объект {cluster[0]}")
        else:
            print(f"  {position}. Группа {cluster}")
        position += 1

    print(f"\nРанги: {result['hon_ranking']}")
    print(f"tau: {compute_my_kendall(result['hon_bin_result_matrix'], target_matrix):.3f}")

    print()