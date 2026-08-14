import numpy as np

from utility import saver
from utility.generator import matrix_to_ranks, matrix_to_ordering, compute_my_kendall, ranks_to_matrix, \
    get_best_tolerance


def find_weights_matrix_method(expert_matrices: list[np.ndarray],
                               target_matrix: np.ndarray) -> (np.ndarray, np.ndarray):

    target_vec = target_matrix.flatten()
    expert_vecs = np.array([expert.flatten() for expert in expert_matrices]).T

    w, _, _, _ = np.linalg.lstsq(expert_vecs, target_vec, rcond=None)
    weights = np.round(w, 3)
    result_matrix = np.sum([weights[i] * expert_matrices[i] for i in range(len(expert_matrices))], axis=0)

    return weights, result_matrix


def find_weights_rank_method(expert_matrices: list[np.ndarray],
                             target_matrix: np.ndarray) -> (np.ndarray, np.ndarray):

    target_ranks = target_matrix.sum(axis=1)
    expert_ranks = np.array([expert.sum(axis=1) for expert in expert_matrices]).T

    w, _, _, _ = np.linalg.lstsq(expert_ranks, target_ranks, rcond=None)
    weights = np.round(w.flatten(), 3)

    result_ranks = expert_ranks @ weights

    return weights, result_ranks


def find_weights(target_matrix: np.ndarray, expert_matrices: list[np.ndarray], method: str = 'matrix',
                 tolerance: float = 0.6) -> dict:
    """
    Агрегирует экспертные предпочтения в единое отношение.
    """
    if method == 'matrix':
        weights, result_matrix = find_weights_matrix_method(expert_matrices, target_matrix)

        best_tol = get_best_tolerance(result_matrix, target_matrix)

        hon_ordering = matrix_to_ordering(result_matrix, tolerance)
        ideal_ordering = matrix_to_ordering(result_matrix, best_tol)

        hon_ranking = matrix_to_ranks(result_matrix, tolerance)
        ideal_ranking = matrix_to_ranks(result_matrix, best_tol)

        error = np.linalg.norm(target_matrix-result_matrix, 'fro')

        hbin_result_matrix = ranks_to_matrix(hon_ranking)
        ibin_result_matrix = ranks_to_matrix(ideal_ranking)


        return {
            'weights': weights,
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
        weights, result_ranks = find_weights_rank_method(expert_matrices, target_matrix)

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

    print(f"\nРанги: {matrix_to_ranks(target_matrix,base_tolerance)} \n")

    if n >= m:
        print(f"\nРанговый метод не имеет предсказательной силы из-за недоопределенности!\n")
    if n >= m*m:
        print(f"Матричный метод также не имеет предсказательной силы!\n")


    for method in ['matrix','rank']:
        print(f"{'=' * 40}")
        print(f"Метод: {'Матричный' if method == 'matrix' else 'Ранговый'}")
        print(f"{'=' * 40}")

        result = find_weights(target_matrix, expert_matrices, method=method, tolerance=base_tolerance)
        print("\nВеса экспертов:")
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
        print(f"tau: {compute_my_kendall(result['ideal_bin_result_matrix'],target_matrix):.3f}")
