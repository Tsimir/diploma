from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau, spearmanr


def calculate_agreement_matrix(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Вычисляет матрицу согласованности между экспертами

    Parameters:
    matrices - список матриц предпочтений (каждая n x n)

    Returns:
    agreement_matrix - матрица согласованности размером m x m, где m - количество экспертов
    """
    m = len(matrices)
    agreement_matrix = np.zeros((m, m))

    # Преобразуем матрицы в ранговые столбцы
    rank_columns = []
    for matrix in matrices:
        rank_column = matrix_to_column_rank(matrix).flatten()
        rank_columns.append(rank_column)

    # Вычисляем согласованность между каждой парой экспертов
    for i in range(m):
        for j in range(m):
            if i == j:
                agreement_matrix[i, j] = 1.0  # полная согласованность с самим собой
            else:
                # Используем коэффициент корреляции Спирмена
                corr, _ = spearmanr(rank_columns[i], rank_columns[j])
                agreement_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return agreement_matrix


def calculate_kendall_agreement(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Вычисляет матрицу согласованности с использованием коэффициента Кендалла
    """
    m = len(matrices)
    agreement_matrix = np.zeros((m, m))

    # Преобразуем матрицы в ранговые столбцы
    rank_columns = []
    for matrix in matrices:
        rank_column = matrix_to_column_rank(matrix).flatten()
        rank_columns.append(rank_column)

    for i in range(m):
        for j in range(m):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                corr, _ = kendalltau(rank_columns[i], rank_columns[j])
                agreement_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return agreement_matrix


def plot_agreement_heatmap(agreement_matrix: np.ndarray,
                           expert_names: Optional[List[str]] = None,
                           title: str = "Матрица согласованности экспертов",
                           cmap: str = "RdYlBu",
                           method: str = "spearman",
                           figsize: tuple = (12, 10)):
    """
    Рисует тепловую карту согласованности экспертов

    Parameters:
    agreement_matrix - матрица согласованности
    expert_names - список имен экспертов
    title - заголовок графика
    cmap - цветовая карта
    method - метод расчета согласованности (для подписи)
    figsize - размер фигуры
    """
    m = agreement_matrix.shape[0]

    if expert_names is None:
        expert_names = [f"Эксперт {i + 1}" for i in range(m)]

    plt.figure(figsize=figsize)

    # Создаем маску для диагональных элементов (чтобы их можно было выделить)
    mask = np.zeros_like(agreement_matrix)
    np.fill_diagonal(mask, 1)

    # Рисуем основную тепловую карту
    ax = sns.heatmap(agreement_matrix,
                     mask=None,
                     annot=True,
                     fmt=".3f",
                     cmap=cmap,
                     center=0,
                     vmin=-1,
                     vmax=1,
                     xticklabels=expert_names,
                     yticklabels=expert_names,
                     cbar_kws={'label': f'Коэффициент согласованности ({method})'})

    # Выделяем диагональные элементы
    for i in range(m):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Эксперты', fontsize=12)
    plt.ylabel('Эксперты', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Добавляем информацию о среднем уровне согласованности
    mean_agreement = np.mean(agreement_matrix[np.triu_indices(m, k=1)])
    plt.figtext(0.5, 0.01,
                f"Средняя согласованность между экспертами: {mean_agreement:.3f}",
                ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    plt.tight_layout()
    return ax


def analyze_expert_agreement(matrices: List[np.ndarray],
                             expert_names: Optional[List[str]] = None,
                             method: str = "spearman") -> dict:
    """
    Полный анализ согласованности экспертов

    Returns:
    dict с результатами анализа
    """
    if method == "spearman":
        agreement_matrix = calculate_agreement_matrix(matrices)
    elif method == "kendall":
        agreement_matrix = calculate_kendall_agreement(matrices)
    else:
        raise ValueError("Метод должен быть 'spearman' или 'kendall'")

    m = len(matrices)

    # Анализ согласованности
    upper_tri_indices = np.triu_indices(m, k=1)
    pairwise_agreements = agreement_matrix[upper_tri_indices]

    analysis = {
        'agreement_matrix': agreement_matrix,
        'mean_agreement': np.mean(pairwise_agreements),
        'std_agreement': np.std(pairwise_agreements),
        'min_agreement': np.min(pairwise_agreements),
        'max_agreement': np.max(pairwise_agreements),
        'method': method
    }

    # Анализ согласованности для каждого эксперта
    expert_analysis = []
    for i in range(m):
        # Согласованность с другими экспертами (исключая самого себя)
        other_agreements = np.delete(agreement_matrix[i], i)
        expert_analysis.append({
            'expert': expert_names[i] if expert_names else f"Эксперт {i + 1}",
            'mean_with_others': np.mean(other_agreements),
            'std_with_others': np.std(other_agreements),
            'min_with_others': np.min(other_agreements),
            'max_with_others': np.max(other_agreements)
        })

    analysis['expert_analysis'] = expert_analysis

    return analysis


def print_agreement_analysis(analysis: dict):
    """Печатает результаты анализа согласованности"""
    print("=" * 60)
    print("АНАЛИЗ СОГЛАСОВАННОСТИ ЭКСПЕРТОВ")
    print("=" * 60)
    print(f"Метод расчета: {analysis['method']}")
    print(f"Количество экспертов: {len(analysis['expert_analysis'])}")
    print(f"Средняя согласованность: {analysis['mean_agreement']:.3f}")
    print(f"Стандартное отклонение: {analysis['std_agreement']:.3f}")
    print(f"Минимальная согласованность: {analysis['min_agreement']:.3f}")
    print(f"Максимальная согласованность: {analysis['max_agreement']:.3f}")

    print("\n" + "-" * 60)
    print("АНАЛИЗ ПО ЭКСПЕРТАМ:")
    print("-" * 60)
    for expert_info in analysis['expert_analysis']:
        print(f"{expert_info['expert']}:")
        print(f"  Средняя согласованность с другими: {expert_info['mean_with_others']:.3f}")
        print(f"  Стандартное отклонение: {expert_info['std_with_others']:.3f}")
        print(f"  Диапазон: [{expert_info['min_with_others']:.3f}, {expert_info['max_with_others']:.3f}]")
        print()


# Функция из generator.py (для совместимости)
def matrix_to_column_rank(matrix):
    """Ранговый столбец: количество объектов, которые доминирует текущий (включая себя)"""
    return np.sum(matrix, axis=1).reshape(-1, 1)


def main():
    """Основная функция для демонстрации работы"""
    # Импортируем функции для чтения матриц
    try:
        from utility.saver import read_matrices_from_csv
    except ImportError:
        print("Не удалось импортировать read_matrices_from_csv из generator")
        return

    # Чтение матриц из файлов
    try:
        # Матрицы авто продаж
        auto_matrices = read_matrices_from_csv("Оптимизация авто продаж.csv")
        print(f"Загружено матриц авто продаж: {len(auto_matrices)}")

        # Имена экспертов для авто продаж (названия критериев)
        auto_expert_names = [
            "По году выпуска", "По состоянию", "По стоимости покупки", "По работам",
            "По типу кузова", "По объему", "По мощности", "По КПП", "По объявлениям",
            "По предлагаемой цене", "По внешнему виду", "По салону", "По докам",
            "По сложности ремонта", "По стране", "Критерий - выручка",
            "Критерий - интерес онлайн", "Критерий - интерес оффлайн",
            "Критерий - рубледень", "Критерий - скорость продажи"
        ]

        # Анализ согласованности для авто продаж
        print("\n" + "=" * 80)
        print("АНАЛИЗ СОГЛАСОВАННОСТИ ДЛЯ АВТО ПРОДАЖ")
        print("=" * 80)

        # Анализ с коэффициентом Спирмена
        analysis_spearman = analyze_expert_agreement(auto_matrices, auto_expert_names, "spearman")
        print_agreement_analysis(analysis_spearman)

        # Визуализация
        plot_agreement_heatmap(analysis_spearman['agreement_matrix'],
                               auto_expert_names,
                               "Согласованность экспертов по критериям авто продаж (Спирмен)",
                               method="spearman")
        plt.show()

        # Анализ с коэффициентом Кендалла
        analysis_kendall = analyze_expert_agreement(auto_matrices, auto_expert_names, "kendall")
        print_agreement_analysis(analysis_kendall)

        plot_agreement_heatmap(analysis_kendall['agreement_matrix'],
                               auto_expert_names,
                               "Согласованность экспертов по критериям авто продаж (Кендалл)",
                               method="kendall")
        plt.show()

    except FileNotFoundError:
        print("Файл 'Оптимизация авто продаж.csv' не найден")

    # Анализ сгенерированных матриц
    try:
        direct_matrices = read_matrices_from_csv('../direct_matrices.csv')
        print(f"\nЗагружено сгенерированных матриц: {len(direct_matrices)}")

        if len(direct_matrices) > 0:
            # Анализ сгенерированных матриц
            analysis_direct = analyze_expert_agreement(direct_matrices, method="spearman")
            print_agreement_analysis(analysis_direct)

            plot_agreement_heatmap(analysis_direct['agreement_matrix'],
                                   title="Согласованность сгенерированных экспертов (Спирмен)",
                                   method="spearman")
            plt.show()

    except FileNotFoundError:
        print("Файл 'direct_matrices.csv' не найден")


if __name__ == "__main__":
    main()