import numpy as np

from utility.generator import compute_my_spearman, matrix_to_ordering, compute_my_kendall, matrix_to_ranks, \
    get_best_tolerance, \
    ranks_to_matrix
from utility.saver import read_matrices_from_csv


def order_by_preferences(PREF:np.ndarray, tolerance: float = 0.6) -> (list[list[int]], np.ndarray):
    """Жадный алгоритм Order-By-Preferences из статьи Коэна"""
    n = PREF.shape[0]
    V = list(range(n))
    pi = np.zeros(n)
    for v in V:
        pi[v] = np.sum(PREF[v, :]) - np.sum(PREF[:, v])

    scores = np.zeros(n)
    current_rank = 1

    while V:
        t = max(V, key=lambda v: pi[v])
        scores[t] = current_rank
        V.remove(t)
        current_rank += 1

        for v in V:
            pi[v] += PREF[t, v] - PREF[v, t]

    sorted_indices = np.argsort(scores)

    clusters = []

    i = 0
    while i < n:
        current_cluster = [int(sorted_indices[i] + 1)]
        current_score = scores[sorted_indices[i]]

        j = i + 1
        while j < n and abs(scores[sorted_indices[j]] - current_score) < tolerance:
            current_cluster.append(int(sorted_indices[j] + 1))
            j += 1

        clusters.append(current_cluster)
        i = j

    return clusters


def loss_function(R:np.ndarray, feedback:list[tuple[int, int]]) -> float:
    """Loss(R,F) = (1/|F|) * sum_{(u,v) in F} (1 - R(u,v))"""
    if len(feedback) == 0:
        return 0
    total_loss = 0
    for u, v in feedback:
        total_loss += 1 - R[u, v]
    return total_loss / len(feedback)


def hedge_algorithm(expert_matrices:list[np.ndarray],
                    feedback_sequence:list[list[tuple[int, int]]], beta:float=0.5) -> (np.ndarray, np.ndarray):
    """
    Алгоритм Hedge.

    Параметры:
        expert_matrices: список матриц экспертов
        feedback_sequence: СПИСОК МНОЖЕСТВ обратной связи для каждого раунда
        beta: параметр алгоритма (0 < beta < 1)
    """
    m = len(expert_matrices)
    n = expert_matrices[0].shape[0]

    weights = np.ones(m) / m

    # Для каждого раунда
    for t, feedback in enumerate(feedback_sequence):
        # Считаем потери экспертов на этом раунде
        losses = np.zeros(m)
        for i in range(m):
            losses[i] = loss_function(expert_matrices[i], feedback)

        # Обновляем веса: w_i = w_i * beta^{loss_i}
        weights = weights * (beta ** losses)

        # Нормализация
        weights = weights / np.sum(weights)

    # Строим итоговую матрицу как взвешенную сумму
    result_matrix = np.zeros((n, n))
    for i in range(m):
        result_matrix += weights[i] * expert_matrices[i]

    return weights, result_matrix


def find_weights(target_matrix: np.ndarray, expert_matrices: list[np.ndarray],
                 beta:float=0.5, n_rounds:int=10, tolerance: float = 0.5) -> dict:
    """
    Агрегация методом Коэна с несколькими раундами обучения.
    """
    m = target_matrix.shape[0]

    all_feedback = []
    for i in range(m):
        for j in range(m):
            if i != j and target_matrix[i, j] == 1:
                all_feedback.append((i, j))

    np.random.shuffle(all_feedback)

    round_size = max(1, len(all_feedback) // n_rounds)
    feedback_sequence = []
    for i in range(0, len(all_feedback), round_size):
        feedback_sequence.append(all_feedback[i:i + round_size])

    weights, result_matrix = hedge_algorithm(
        expert_matrices,
        feedback_sequence,
        beta=beta
    )

    best_tol = get_best_tolerance(result_matrix, target_matrix)

    hon_ordering = order_by_preferences(result_matrix, tolerance)
    ideal_ordering = order_by_preferences(result_matrix, best_tol)

    hon_ranking = matrix_to_ranks(result_matrix, tolerance)
    ideal_ranking = matrix_to_ranks(result_matrix, best_tol)

    hbin_result_matrix = ranks_to_matrix(hon_ranking)
    ibin_result_matrix = ranks_to_matrix(ideal_ranking)

    error = np.linalg.norm(target_matrix - result_matrix, 'fro')

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


if __name__ == "__main__":
    all_matrices = read_matrices_from_csv("../test_matrices/control_matrices.csv")

    n = len(all_matrices) - 1
    m = len(all_matrices[0])
    base_tolerance = 0.6

    np.random.seed(0)

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

    n_rounds = 5

    for beta in [0.5]:
        print(f"{'=' * 40}")
        print(f"Метод: Коэна с обучением, beta={beta}")
        print(f"{'=' * 40}")

        result = find_weights(target_matrix, expert_matrices,
                              beta=beta, n_rounds=n_rounds, tolerance=base_tolerance)

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
        print(f"tau: {compute_my_kendall(result['ideal_bin_result_matrix'], target_matrix):.3f}")
        print(
            f"rho: {compute_my_spearman(result['ideal_ranking'], matrix_to_ranks(target_matrix, base_tolerance)):.3f}")

        print()