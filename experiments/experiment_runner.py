import csv
import time
import itertools
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

# Импорт модулей проекта
from utility.generator import (
    PrefMaGen_direct, PrefMaGen_level, PrefMaGen_noise,
    matrix_to_ranks, matrix_to_ordering, compute_my_kendall,
    compute_my_spearman
)


from methods.agregate_graph import find_weights as graph_method
from methods.naive_svd import find_weights as bulatov_method
from methods.greedy_obp import find_weights as cohen_method
from methods.conical_mehods import find_weights as cone_method

# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА (меняйте здесь нужные значения)
# ============================================================================

# Параметры сетки
M_VALUES = [100, 200, 300, 500, 1000] # list(range(100, 2100, 100))  количество объектов
N_EXPERTS_VALUES = [100] # list(range(100, 2100, 100))   количество экспертов
GEN_METHODS = ['level','noise']  # методы генерации
NOISE_LEVELS = [0.2]  # уровень шума (для noise)
P_DOMINATE_VALUES = [0.3]  # вероятность доминирования (для direct)

# Параметры метода Коэна
BETA_VALUES = [0.5]  # скорость адаптации
N_ROUNDS_VALUES = [5]  # число раундов обучения

# Какие методы запускать (можно закомментировать ненужные)
METHODS_TO_RUN = [
    'graph_method'
]

# Количество повторных запусков для каждой комбинации параметров
RUNS_PER_COMBO = 10

# Базовый seed для воспроизводимости
BASE_SEED = 3806

# Путь для сохранения результатов (измените при необходимости)
OUTPUT_DIR = "../experiment_output"
# Имя выходного CSV-файла
OUTPUT_FILENAME = "./graph_pref_restoration_results.csv"

# Порог для "честных" результатов (фиксированный tolerance)
HONEST_TOLERANCE_BULL = 0.6
HONEST_TOLERANCE_KUZN = 0.0

# Шаг перебора tolerance при поиске идеального (чем меньше, тем точнее, но дольше)
TOLERANCE_STEP = 0.1

# Показывать ли прогресс-бар
SHOW_PROGRESS = True

# Количество параллельных процессов (None = число ядер CPU)
N_WORKERS = 5

CHECKPOINT_EVERY = 5

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def generate_matrices_for_experiment(
        m: int,
        n_experts: int,
        gen_method: str,
        noise_level: float,
        p_dominate: float,
        seed: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Генерирует экспертные матрицы и целевую матрицу.
    Возвращает (list_expert_matrices, target_matrix).
    """
    np.random.seed(seed)

    # Генерируем целевую матрицу (без шума) в зависимости от метода
    if gen_method == 'direct':
        target = PrefMaGen_direct(m, p_dominate)
    elif gen_method == 'level':
        target = PrefMaGen_level(m)
    elif gen_method == 'noise':
        # Для noise: сначала генерируем базовый линейный порядок, затем зашумляем его
        target = PrefMaGen_level(m)
    else:
        raise ValueError(f"Unknown gen_method: {gen_method}")

    # Генерируем экспертные матрицы
    experts = []
    for _ in range(n_experts):
        if gen_method == 'direct':
            mat = PrefMaGen_direct(m, p_dominate)
        elif gen_method == 'level':
            mat = PrefMaGen_level(m)
        elif gen_method == 'noise':
            mat = PrefMaGen_noise(target, noise_level)
        experts.append(mat)

    return experts, target


# Общая функция генерации данных, учитывающая метод
def generate_data(
        m: int,
        n_experts: int,
        gen_method: str,
        noise_level: float,
        p_dominate: float,
        seed: int
) -> Tuple[List[np.ndarray], np.ndarray]:

    np.random.seed(seed)
    if gen_method == 'noise':
        target = PrefMaGen_level(m)
        experts = [PrefMaGen_noise(target, noise_level) for _ in range(n_experts)]
    elif gen_method == 'direct':
        target = PrefMaGen_direct(m, p_dominate)
        experts = [PrefMaGen_direct(m, p_dominate) for _ in range(n_experts)]
    elif gen_method == 'level':
        target = PrefMaGen_level(m)
        experts = [PrefMaGen_level(m) for _ in range(n_experts)]
    else:
        raise ValueError(f"Unsupported gen_method: {gen_method}")
    return experts, target


def compute_pair_accuracy(Z_pred: np.ndarray, Z_true: np.ndarray) -> float:
    """Доля пар (i,j), i<j, где направление предпочтения совпадает."""
    m = Z_pred.shape[0]
    correct = 0
    total = 0
    for i in range(m):
        for j in range(i + 1, m):
            true_dir = Z_true[i, j] - Z_true[j, i]
            pred_dir = Z_pred[i, j] - Z_pred[j, i]
            if true_dir == 0:
                # в истинном отношении объекты эквивалентны – считаем предсказание верным, если тоже 0
                if pred_dir == 0:
                    correct += 1
            else:
                if true_dir * pred_dir > 0:
                    correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def run_method(
        method_name: str,
        target_matrix: np.ndarray,
        expert_matrices: List[np.ndarray],
        beta: float,
        n_rounds: int,
        seed: int
) -> Optional[Dict[str, Any]]:
    """Запускает один метод агрегации и возвращает словарь с метриками."""
    np.random.seed(seed)  # для воспроизводимости внутри методов (например, Cohen)

    try:
        if method_name == 'Bulatov_matrix':
            # Проверка применимости: m^2 >= n
            m = target_matrix.shape[0]
            n = len(expert_matrices)
            if n > m * m:
                return None  # условие не выполнено
            res = bulatov_method(target_matrix, expert_matrices, method='matrix', tolerance=HONEST_TOLERANCE_BULL)
            weights = res.get('weights', None)
        elif method_name == 'Bulatov_rank':
            m = target_matrix.shape[0]
            n = len(expert_matrices)
            if n > m:
                return None
            res = bulatov_method(target_matrix, expert_matrices, method='rank', tolerance=HONEST_TOLERANCE_BULL)
            weights = res.get('weights', None)
        elif method_name == 'Cohen':
            res = cohen_method(target_matrix, expert_matrices, beta=beta, n_rounds=n_rounds, tolerance=HONEST_TOLERANCE_BULL)
            weights = res.get('weights', None)
        elif method_name == 'cone_direct':
            res = cone_method(target_matrix, expert_matrices, method='cone_direct', tolerance=HONEST_TOLERANCE_KUZN)
            weights = None  # прямой конусный метод не возвращает веса
        elif method_name == 'cone_reg':
            res = cone_method(target_matrix, expert_matrices, method='cone_reg', tolerance=HONEST_TOLERANCE_KUZN)
            weights = res.get('weights', None)
        elif method_name == 'graph_method':
            res = graph_method(
                expert_matrices,
                target_matrix=target_matrix,
                tolerance=HONEST_TOLERANCE_BULL
            )
            weights = res.get('weights', None)
        else:
            return None
    except Exception as e:
        print(f"Ошибка в методе {method_name}: {e}", file=sys.stderr)
        return None

    # Извлекаем результаты
    hon_ranking = res.get('hon_ranking')
    ideal_ranking = res.get('ideal_ranking')
    hon_bin = res.get('hon_bin_result_matrix')
    ideal_bin = res.get('ideal_bin_result_matrix')
    error = res.get('error', np.nan)

    if hon_ranking is None or ideal_ranking is None or hon_bin is None or ideal_bin is None:
        return None

    # Ранги целевой матрицы (для scipy метрик)
    target_ranks = matrix_to_ranks(target_matrix, HONEST_TOLERANCE_BULL)
    target_ordering = matrix_to_ordering(target_matrix, HONEST_TOLERANCE_BULL)

    # Подготовка словаря с результатами
    def compute_metrics(ranking, bin_matrix, prefix):
        # Собственные корреляции
        my_kendall = compute_my_kendall(bin_matrix, target_matrix)
        my_spearman = compute_my_spearman(ranking, target_ranks)
        # Scipy корреляции
        scipy_kendall, _ = kendalltau(ranking, target_ranks)
        scipy_spearman, _ = spearmanr(ranking, target_ranks)
        # Точность попарных сравнений
        pair_acc = compute_pair_accuracy(bin_matrix, target_matrix)
        # Упорядочение (кластеры)
        ordering = matrix_to_ordering(bin_matrix, HONEST_TOLERANCE_BULL)  # для честных используем тот же tolerance
        ordering_str = str(ordering)
        return {
            f'{prefix}_my_kendall': my_kendall,
            f'{prefix}_my_spearman': my_spearman,
            f'{prefix}_scipy_kendall': scipy_kendall,
            f'{prefix}_scipy_spearman': scipy_spearman,
            f'{prefix}_pair_accuracy': pair_acc,
            f'{prefix}_ordering': ordering_str
        }

    hon_metrics = compute_metrics(hon_ranking, hon_bin, 'hon')
    ideal_metrics = compute_metrics(ideal_ranking, ideal_bin, 'ideal')

    # Преобразование весов в строку
    if weights is not None:
        weights_str = ','.join([f"{w:.4f}" for w in weights])
    else:
        weights_str = "None"

    # Целевое упорядочение
    target_ordering_str = str(target_ordering)

    result = {
        'frobenius_error': error,
        'weights': weights_str,
        'target_ordering': target_ordering_str,
        **hon_metrics,
        **ideal_metrics
    }
    return result


def single_experiment(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Один эксперимент: генерация данных + прогон всех методов.
    params содержит все необходимые параметры.
    Возвращает список результатов (по одному на метод) или пустой список.
    """
    m = params['m']
    n_experts = params['n_experts']
    gen_method = params['gen_method']
    noise_level = params['noise_level']
    p_dominate = params['p_dominate']
    beta = params['beta']
    n_rounds = params['n_rounds']
    run_id = params['run']
    seed = params['seed']
    methods = params['methods_to_run']

    # Генерируем данные (одинаковые для всех методов в этом run)
    experts, target = generate_data(m, n_experts, gen_method, noise_level, p_dominate, seed)

    results_list = []
    for method in methods:
        start_time = time.time()
        metrics = run_method(
            method, target, experts, beta, n_rounds, seed + hash(method) % 10000
        )
        elapsed = time.time() - start_time
        if metrics is None:
            continue  # метод не применим или ошибка

        # Формируем полную запись
        record = {
            'm': m,
            'n_experts': n_experts,
            'gen_method': gen_method,
            'noise_level': noise_level if gen_method == 'noise' else '',
            'p_dominate': p_dominate if gen_method == 'direct' else '',
            'beta': beta if method == 'Cohen' else '',
            'n_rounds': n_rounds if method == 'Cohen' else '',
            'run': run_id,
            'seed': seed,
            'method': method,
            'time_seconds': elapsed,
            **metrics
        }
        results_list.append(record)

    return results_list


def generate_all_combinations() -> List[Dict[str, Any]]:
    """Генерирует все комбинации параметров для экспериментов."""
    combos = []
    run_counter = 0
    for m in M_VALUES:
        for n_exp in N_EXPERTS_VALUES:
            for gen_method in GEN_METHODS:
                # Определяем наборы параметров в зависимости от метода генерации
                if gen_method == 'direct':
                    p_list = P_DOMINATE_VALUES
                    noise_list = [0.0]  # фиктивное значение
                elif gen_method == 'level':
                    p_list = [0.0]
                    noise_list = [0.0]
                elif gen_method == 'noise':
                    p_list = [0.0]
                    noise_list = NOISE_LEVELS
                else:
                    continue

                for p_dom in p_list:
                    for noise in noise_list:
                        # Для метода Cohen свои параметры
                        for beta in BETA_VALUES:
                            for n_rounds in N_ROUNDS_VALUES:
                                for run_id in range(RUNS_PER_COMBO):
                                    seed = BASE_SEED + run_counter * 1000 + run_id
                                    combo = {
                                        'm': m,
                                        'n_experts': n_exp,
                                        'gen_method': gen_method,
                                        'noise_level': noise,
                                        'p_dominate': p_dom,
                                        'beta': beta,
                                        'n_rounds': n_rounds,
                                        'run': run_id,
                                        'seed': seed,
                                        'methods_to_run': METHODS_TO_RUN
                                    }
                                    combos.append(combo)
                                    run_counter += 1
    # Для сокращения числа экспериментов можно ограничить, но оставим как есть
    return combos


def save_results_to_csv(results: List[Dict[str, Any]], output_path: Path):
    """Сохраняет список словарей в CSV."""
    if not results:
        print("Нет результатов для сохранения.")
        return

    # Определяем все возможные ключи
    fieldnames = [
        'm', 'n_experts', 'gen_method', 'noise_level', 'p_dominate',
        'beta', 'n_rounds', 'run', 'seed', 'method', 'time_seconds',
        'hon_ordering', 'hon_my_kendall', 'hon_my_spearman', 'hon_scipy_kendall',
        'hon_scipy_spearman', 'hon_pair_accuracy',
        'ideal_ordering', 'ideal_my_kendall', 'ideal_my_spearman', 'ideal_scipy_kendall',
        'ideal_scipy_spearman', 'ideal_pair_accuracy',
        'frobenius_error', 'weights', 'target_ordering'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Сохранено {len(results)} записей в {output_path}")


def main():
    print("Генерация всех комбинаций параметров...")
    all_combos = generate_all_combinations()
    print(f"Всего экспериментов (запусков методов): {len(all_combos) * len(METHODS_TO_RUN)}")
    print(f"Комбинаций параметров (без учёта методов): {len(all_combos)}")

    # Параллельное выполнение
    import multiprocessing as mp
    n_workers = N_WORKERS or mp.cpu_count()
    print(f"Используется процессов: {n_workers}")


    with mp.Pool(processes=n_workers) as pool:
        if SHOW_PROGRESS:
            results_iter = tqdm(pool.imap_unordered(single_experiment, all_combos), total=len(all_combos),
                                desc="Эксперименты")
        else:
            results_iter = pool.imap_unordered(single_experiment, all_combos)

        all_records = []
        checkpoint_counter = 0  # ← ДОБАВИТЬ

        for records in results_iter:
            if records:
                all_records.extend(records)
                checkpoint_counter += len(records)  # ← ДОБАВИТЬ

                # Промежуточное сохранение ← ДОБАВИТЬ ЭТОТ БЛОК
                if checkpoint_counter >= CHECKPOINT_EVERY:
                    temp_path = Path(OUTPUT_DIR) / f"{OUTPUT_FILENAME}.tmp"
                    save_results_to_csv(all_records, temp_path)
                    checkpoint_counter = 0
                    print(f"\n[Checkpoint] Сохранено {len(all_records)} записей")

    # Сохраняем финальный результат
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILENAME
    save_results_to_csv(all_records, output_path)

    # Удаляем временный файл ← ДОБАВИТЬ
    temp_path = Path(OUTPUT_DIR) / f"{OUTPUT_FILENAME}.tmp"
    if temp_path.exists():
        temp_path.unlink()

    print("Готово!")

if __name__ == "__main__":
    main()