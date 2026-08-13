import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, kendalltau


def semkin_similarity(x_i, x_j, tau=0.0, eta=-1.0):
    """
    Простая реализация меры сходства Сёмкина
    """
    # Преобразуем в множества
    A = set(x_i)
    B = set(x_j)

    intersection = len(A & B)
    len_A = len(A)
    len_B = len(B)

    if intersection == 0:
        return 0.0

    # Вычисляем K_τ компоненты
    def K_tau(n_intersect, n_total, tau_param):
        if n_total == 0:
            return 0.0
        denominator = (1 + tau_param) * n_total - tau_param * n_intersect
        return n_intersect / denominator if denominator > 0 else 0.0

    K_ij = K_tau(intersection, len_A, tau)
    K_ji = K_tau(intersection, len_B, tau)

    # Специальные случаи
    if eta == -np.inf:  # Браун-Бланке
        return intersection / max(len_A, len_B)
    elif eta == np.inf:  # Симпсон
        return intersection / min(len_A, len_B)
    elif eta == 0:  # Геометрическое среднее
        return np.sqrt(K_ij * K_ji)
    else:
        # Общий случай
        if K_ij == 0 or K_ji == 0:
            return 0.0 if eta < 0 else (K_ij ** eta + K_ji ** eta) / 2
        return ((K_ij ** eta + K_ji ** eta) / 2) ** (1 / eta)


# Стандартные коэффициенты как отдельные функции
def jaccard(x, y):
    return semkin_similarity(x, y, tau=1.0, eta=-1.0)


def sorensen(x, y):
    return semkin_similarity(x, y, tau=0.0, eta=-1.0)


def simpson(x, y):
    return semkin_similarity(x, y, tau=0.0, eta=np.inf)


def ochiai(x, y):
    return semkin_similarity(x, y, tau=0.0, eta=2.0)


# Коэффициенты корреляции из scipy
def spearman(x, y):
    """Коэффициент корреляции Спирмена"""
    if len(x) != len(y):
        raise ValueError("Векторы должны быть одинаковой длины")
    corr, p_value = spearmanr(x, y)
    return corr


def kendall(x, y):
    """Коэффициент корреляции Кендалла"""
    if len(x) != len(y):
        raise ValueError("Векторы должны быть одинаковой длины")
    corr, p_value = kendalltau(x, y)
    return corr


def compare_matrices(matrix1, matrix2, method=jaccard, compare_type='flatten'):
    """
    Сравниваем две матрицы выбранным методом
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Матрицы должны быть одного размера")

    if compare_type == 'flatten':
        return method(matrix1.flatten(), matrix2.flatten())
    elif compare_type == 'rowwise':
        similarities = [method(matrix1[i], matrix2[i]) for i in range(matrix1.shape[0])]
        return np.mean(similarities)
    elif compare_type == 'columnwise':
        similarities = [method(matrix1[:, j], matrix2[:, j]) for j in range(matrix1.shape[1])]
        return np.mean(similarities)


def plot_matrix_correlations(matrices, names=None, method=jaccard):
    """
    Рисуем heatmap корреляций между матрицами
    """
    n = len(matrices)
    if names is None:
        names = [f'M{i + 1}' for i in range(n)]

    # Вычисляем матрицу сходств
    corr_matrix = np.eye(n)  # Диагональ = 1

    for i in range(n):
        for j in range(i + 1, n):
            sim = compare_matrices(matrices[i], matrices[j], method)
            corr_matrix[i, j] = sim
            corr_matrix[j, i] = sim

    # Рисуем
    method_name = method.__name__ if hasattr(method, '__name__') else str(method)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix,
                xticklabels=names,
                yticklabels=names,
                annot=True,
                cmap='viridis',
                vmin=0, vmax=1)
    plt.title(f'Матрица сходств ({method_name})')
    plt.tight_layout()
    plt.show()

    return corr_matrix


# Пример использования
if __name__ == "__main__":
    print("=== Простой пример использования ===\n")

    # Тестовые данные для коэффициентов корреляции
    print("=== Коэффициенты корреляции ===")
    x = [1, 2, 3, 4, 5]
    y1 = [1, 2, 3, 4, 5]  # Полная корреляция
    y2 = [5, 4, 3, 2, 1]  # Отрицательная корреляция
    y3 = [1, 3, 2, 5, 4]  # Частичная корреляция

    print(f"X:  {x}")
    print(f"Y1: {y1} (полная корреляция)")
    print(f"Y2: {y2} (отрицательная корреляция)")
    print(f"Y3: {y3} (частичная корреляция)")
    print()

    print("Коэффициенты Спирмена:")
    print(f"X-Y1: {spearman(x, y1):.3f}")
    print(f"X-Y2: {spearman(x, y2):.3f}")
    print(f"X-Y3: {spearman(x, y3):.3f}")
    print()

    print("Коэффициенты Кендалла:")
    print(f"X-Y1: {kendall(x, y1):.3f}")
    print(f"X-Y2: {kendall(x, y2):.3f}")
    print(f"X-Y3: {kendall(x, y3):.3f}")

    print("\n" + "=" * 50)

    # Тестовые данные для мер сходства
    print("=== Меры сходства Сёмкина ===")
    obj1 = [1, 2, 3, 4, 5]
    obj2 = [3, 4, 5, 6, 7]

    print(f"Объект 1: {obj1}")
    print(f"Объект 2: {obj2}")
    print()

    # Сравниваем разными методами
    print("Сходство объектов:")
    print(f"Жаккар:    {jaccard(obj1, obj2):.3f}")
    print(f"Сёренсен:  {sorensen(obj1, obj2):.3f}")
    print(f"Симпсон:   {simpson(obj1, obj2):.3f}")
    print(f"Охмана:    {ochiai(obj1, obj2):.3f}")
    print(f"Сёмкин(τ=0.5,η=0.5): {semkin_similarity(obj1, obj2, 0.5, 0.5):.3f}")

    print("\n" + "=" * 50)

    # С матрицами
    print("=== Сравнение матриц ===")
    np.random.seed(42)
    A = np.random.randint(0, 10, (3, 3))
    B = A + np.random.randint(0, 2, (3, 3))  # Похожая матрица
    C = np.random.randint(0, 10, (3, 3))  # Случайная

    print("Матрица A:")
    print(A)
    print("\nМатрица B (похожая на A):")
    print(B)
    print("\nМатрица C (случайная):")
    print(C)

    print(f"\nСходство матриц (разные методы):")
    print(f"Жаккар:    A-B: {compare_matrices(A, B, jaccard):.3f}, A-C: {compare_matrices(A, C, jaccard):.3f}")
    print(f"Спирмен:   A-B: {compare_matrices(A, B, spearman):.3f}, A-C: {compare_matrices(A, C, spearman):.3f}")
    print(f"Кендалл:   A-B: {compare_matrices(A, B, kendall):.3f}, A-C: {compare_matrices(A, C, kendall):.3f}")

    # Heatmap для нескольких матриц
    matrices = [A, B, C]
    names = ['A', 'B', 'C']

    print("\nСтроим heatmap корреляций (Спирмен)...")
    plot_matrix_correlations(matrices, names, spearman)

    print("\nСтроим heatmap корреляций (Кендалл)...")
    plot_matrix_correlations(matrices, names, kendall)