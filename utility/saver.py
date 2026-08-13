import csv
import os

import numpy as np

from utility.generator import PrefMaGen_direct, PrefMaGen_level, PrefMaGen_noise


def save_matrices_to_csv(matrices, filename):
    """Сохраняет список матриц в CSV файл"""
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Записываем заголовок с размерами
        if matrices:
            n = len(matrices[0])  # размерность матрицы
            m = len(matrices)  # количество матриц
            csvwriter.writerow([f'n={n}', f'm={m}'])

        # Записываем все матрицы подряд
        for idx, matrix in enumerate(matrices):
            csvwriter.writerow([f'Matrix_{idx + 1}'])
            for row in matrix:
                csvwriter.writerow(row)
            csvwriter.writerow([])  # пустая строка между матрицами

def read_matrices_from_csv(filename):
    """Читает список матриц из CSV файла"""
    matrices = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        current_matrix = []

        for row in reader:
            if not row:  # пустая строка - конец матрицы
                if current_matrix:
                    matrices.append(np.array(current_matrix, dtype=int))
                    current_matrix = []
                continue

            # Пропускаем заголовочные строки
            if row[0].startswith('Matrix_') or row[0].startswith('n='):
                continue

            # Добавляем строку матрицы
            numeric_row = [int(x) for x in row]
            current_matrix.append(numeric_row)

        # Добавляем последнюю матрицу если есть
        if current_matrix:
            matrices.append(np.array(current_matrix, dtype=int))

    return matrices

def generate_matrices_csv(m,n, method='direct',
                          filename='preference_matrices.csv', p_dominate=0.3, noise_level=0.0,
                          max_iter=100, tolerance = 0.1):
    """
    Генерирует m матриц и сохраняет в CSV файл

    Parameters:
    m - размерность матриц
    n - количество матриц (экспертов)
    method - метод генерации: 'direct', 'level' или 'noise'
    filename - имя файла (полный путь)
    p_dominate - вероятность доминирования (для direct)
    noise_level - уровень шума (для noise)
    """
    matrices = []

    if method == 'noise':
        # Генерируем одну базовую матрицу линейного порядка
        base_matrix = PrefMaGen_level(m)

        # Сохраняем целевую матрицу без шума
        target_matrix = base_matrix.copy()

        # Генерируем n зашумленных матриц
        for _ in range(n):
            if method == 'noise':
                matrix = PrefMaGen_noise(base_matrix, noise_level)
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

    save_matrices_to_csv(matrices, filename)


if __name__ == "__main__":
    n = 30  # количество экспертов
    m = 100  # размерность матриц

    dominate = 0.5
    noise = 0.3

    save_path = "../test_matrices"
    print(f"\nФайлы будут сохранены в: {save_path}")



    # Генерируем матрицы
    generate_matrices_csv(m, n, method='direct', filename="../test_matrices/direct_matrices.csv", p_dominate=dominate)
    generate_matrices_csv(m, n, method='level', filename="../test_matrices/level_matrices.csv")
    generate_matrices_csv(m, n, method='noise', filename="../test_matrices/noise_matrices.csv", noise_level=noise)

    print(f"\nВсе файлы сохранены в: {save_path}")