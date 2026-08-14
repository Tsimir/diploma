import numpy as np
import time

from utility.generator import compute_my_kendall, compute_my_spearman, matrix_to_ranks, PrefMaGen_level, \
    PrefMaGen_noise, PrefMaGen_direct
from methods.naive_svd import find_weights as bulatov_method
from methods.greedy_obp import find_weights as cohen_method
from methods.conical_mehods import find_weights as cone_method
from utility.saver import generate_matrices_csv

m = 21  # количество объектов
n = 20  # количество экспертов

method = 'noise'
tau_wanted = 0.6
noise = 1 - tau_wanted
base_tolerance = 0.6
beta = 0.3
n_rounds = 5

"m^2>n и m>n - хорошо"

print(f"Параметры эксперимента: {n} экспертов, {m} объектов, целевой τ = {tau_wanted}")
print(f"Уровень шума: {noise:.2f}\n")

# Генерация матриц
print("Генерация тестовых данных...")
generate_matrices_csv(m, n, method='level', filename="../test_matrices/level_matrices.csv", noise_level=noise)

matrices = []

if method == 'noise':
    # Генерируем одну базовую матрицу линейного порядка
    base_matrix = PrefMaGen_level(m)

    # Сохраняем целевую матрицу без шума
    target_matrix = base_matrix.copy()

    # Генерируем n зашумленных матриц
    for _ in range(n):
        if method == 'noise':
            matrix = PrefMaGen_noise(base_matrix, noise)
        matrices.append(matrix)

    # Добавляем target В КОНЕЦ
    matrices.append(target_matrix)

else:
    # Генерируем n матриц для экспертов
    for _ in range(n):
        if method == 'direct':
            matrix = PrefMaGen_direct(m, p_dominate)
        elif method == 'level':
            matrix = PrefMaGen_level(m)
        matrices.append(matrix)

    # Создаем target матрицу
    if method == 'direct':
        target_matrix = PrefMaGen_direct(m, p_dominate)
    elif method == 'level':
        target_matrix = PrefMaGen_level(m)

    # Добавляем target в конец
    matrices.append(target_matrix)


all_matrices = matrices
expert_matrices = all_matrices[:-1]
target_matrix = all_matrices[-1]

# Запуск методов
print("\n" + "=" * 70)
print("СРАВНЕНИЕ МЕТОДОВ АГРЕГАЦИИ ЭКСПЕРТНЫХ МНЕНИЙ")
print("=" * 70)

results = {}

print("\n" + "=" * 70)
print("СОГЛАСОВАННОСТЬ ЭКСПЕРТОВ")
print("=" * 70)

# Согласованность каждого эксперта с целевым порядком
expert_taus = []
expert_rhos = []
for i, Z_exp in enumerate(expert_matrices):
    tau = compute_my_kendall(Z_exp, target_matrix)
    rho = compute_my_spearman(matrix_to_ranks(Z_exp), matrix_to_ranks(target_matrix))
    expert_taus.append(tau)
    expert_rhos.append(rho)

print(f"Согласованность экспертов с целевым порядком:")
print(f"  Kendall τ: среднее = {np.mean(expert_taus):.4f} ± {np.std(expert_taus):.4f}")
print(f"  Spearman ρ: среднее = {np.mean(expert_rhos):.4f} ± {np.std(expert_rhos):.4f}")

# Согласованность между экспертами (попарно)
pairwise_taus = []
pairwise_rhos = []
num_exp = len(expert_matrices)
for i in range(num_exp):
    for j in range(i+1, num_exp):
        pairwise_taus.append(compute_my_kendall(expert_matrices[i], expert_matrices[j]))
        pairwise_rhos.append(compute_my_spearman(
            matrix_to_ranks(expert_matrices[i]),
            matrix_to_ranks(expert_matrices[j])
        ))

print(f"\nСогласованность между экспертами (попарно):")
print(f"  Kendall τ: среднее = {np.mean(pairwise_taus):.4f} ± {np.std(pairwise_taus):.4f}")
print(f"  Spearman ρ: среднее = {np.mean(pairwise_rhos):.4f} ± {np.std(pairwise_rhos):.4f}")
print("=" * 70)

# Запуск методов
print("\nЗапуск методов агрегации...")

# 1. Наивный матричный метод
start_time = time.time()
res = bulatov_method(target_matrix, expert_matrices, method='matrix', tolerance=base_tolerance)
execution_time = time.time() - start_time

results['Bulatov (matrix)'] = {
    'hon_tau': compute_my_kendall(res['hon_bin_result_matrix'], target_matrix),
    'hon_rho': compute_my_spearman(res['hon_ranking'], matrix_to_ranks(target_matrix)),
    'ideal_tau': compute_my_kendall(res['ideal_bin_result_matrix'], target_matrix),
    'ideal_rho': compute_my_spearman(res['ideal_ranking'], matrix_to_ranks(target_matrix)),
    'error': res['error'],
    'time': execution_time  # Добавляем время выполнения
}

# 2. Наивный ранговый метод
start_time = time.time()
res = bulatov_method(target_matrix, expert_matrices, method='rank', tolerance=base_tolerance)
execution_time = time.time() - start_time

results['Bulatov (rank)'] = {
    'hon_tau': compute_my_kendall(res['hon_bin_result_matrix'], target_matrix),
    'hon_rho': compute_my_spearman(res['hon_ranking'], matrix_to_ranks(target_matrix)),
    'ideal_tau': compute_my_kendall(res['ideal_bin_result_matrix'], target_matrix),
    'ideal_rho': compute_my_spearman(res['ideal_ranking'], matrix_to_ranks(target_matrix)),
    'error': res['error'],
    'time': execution_time
}

# 3. Метод Коэна
start_time = time.time()
res = cohen_method(target_matrix, expert_matrices, beta=beta, n_rounds=n_rounds, tolerance=base_tolerance)
execution_time = time.time() - start_time

results['Cohen (beta=0.5)'] = {
    'hon_tau': compute_my_kendall(res['hon_bin_result_matrix'], target_matrix),
    'hon_rho': compute_my_spearman(res['hon_ranking'], matrix_to_ranks(target_matrix)),
    'ideal_tau': compute_my_kendall(res['ideal_bin_result_matrix'], target_matrix),
    'ideal_rho': compute_my_spearman(res['ideal_ranking'], matrix_to_ranks(target_matrix)),
    'error': res['error'],
    'time': execution_time
}

# 4. Конусный метод прямой
start_time = time.time()
res = cone_method(target_matrix, expert_matrices, method='cone_direct')
execution_time = time.time() - start_time

results['Kuznets (cone direct)'] = {
    'hon_tau': compute_my_kendall(res['hon_bin_result_matrix'], target_matrix),
    'hon_rho': compute_my_spearman(res['hon_ranking'], matrix_to_ranks(target_matrix)),
    'ideal_tau': compute_my_kendall(res['ideal_bin_result_matrix'], target_matrix),
    'ideal_rho': compute_my_spearman(res['ideal_ranking'], matrix_to_ranks(target_matrix)),
    'error': res['error'],
    'time': execution_time
}

# 5. Конусный метод регуляризованный
start_time = time.time()
res = cone_method(target_matrix, expert_matrices, method='cone_reg')
execution_time = time.time() - start_time

results['Kuznets (cone regul)'] = {
    'hon_tau': compute_my_kendall(res['hon_bin_result_matrix'], target_matrix),
    'hon_rho': compute_my_spearman(res['hon_ranking'], matrix_to_ranks(target_matrix)),
    'ideal_tau': compute_my_kendall(res['ideal_bin_result_matrix'], target_matrix),
    'ideal_rho': compute_my_spearman(res['ideal_ranking'], matrix_to_ranks(target_matrix)),
    'error': res['error'],
    'time': execution_time
}

# Обновим таблицы, добавив колонку времени
print("\n" + "=" * 70)
print("ТАБЛИЦА 1: ЧЕСТНЫЕ РЕЗУЛЬТАТЫ (фиксированная tolerance = 0.6)")
print("=" * 70)
print(f"\n{'Метод':<25} {'Kendall τ':<12} {'Spearman ρ':<12} {'Ошибка':<12} {'Время (с)':<12}")
print("-" * 85)

for method, metrics in results.items():
    print(f"{method:<25} {metrics['hon_tau']:>10.4f}   {metrics['hon_rho']:>10.4f}   {metrics['error']:>10.4f}   {metrics['time']:>10.4f}")

# Таблица 2: Идеальные результаты (с подобранной tolerance)
print("\n" + "=" * 70)
print("ТАБЛИЦА 2: ИДЕАЛЬНЫЕ РЕЗУЛЬТАТЫ (подобранная tolerance)")
print("=" * 70)
print(f"\n{'Метод':<25} {'Kendall τ':<12} {'Spearman ρ':<12}")
print("-" * 70)

for method, metrics in results.items():
    print(f"{method:<25} {metrics['ideal_tau']:>10.4f}   {metrics['ideal_rho']:>10.4f}")

# Лучшие методы
print("\n" + "-" * 70)
print("ЛУЧШИЕ МЕТОДЫ:")
print(f"  По Kendall τ (честный): {max(results, key=lambda x: results[x]['hon_tau'])} ({max(results[x]['hon_tau'] for x in results):.4f})")
print(f"  По Spearman ρ (честный): {max(results, key=lambda x: results[x]['hon_rho'])} ({max(results[x]['hon_rho'] for x in results):.4f})")
print(f"  По Kendall τ (идеальный): {max(results, key=lambda x: results[x]['ideal_tau'])} ({max(results[x]['ideal_tau'] for x in results):.4f})")
print(f"  По Spearman ρ (идеальный): {max(results, key=lambda x: results[x]['ideal_rho'])} ({max(results[x]['ideal_rho'] for x in results):.4f})")
print(f"  По ошибке: {min(results, key=lambda x: results[x]['error'])} ({min(results[x]['error'] for x in results):.4f})")

print("\n" + "=" * 70)