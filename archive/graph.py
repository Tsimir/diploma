
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utility.generator import PrefMaGen_linear, PrefMaGen_level, PrefMaGen_noise, matrix_to_ordering


def visualize_preference_matrix(Z: np.ndarray, title: str = "Граф предпочтений"):
    """
    Визуализирует матрицу предпочтений как ориентированный граф.
    """
    n = Z.shape[0]

    # Создаем ориентированный граф
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Добавляем ребра (только доминирование)
    for i in range(n):
        for j in range(n):
            if i != j and Z[i, j] == 1:
                G.add_edge(i, j)

    # Настройка визуализации
    plt.figure(figsize=(8, 6))

    # Позиционирование вершин
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    labels = {i: str(i + 1) for i in range(n)}

    # Рисуем граф
    nx.draw(G, pos,
            with_labels=True,
            labels=labels,
            node_color='lightblue',
            node_size=500,
            font_size=12,
            font_weight='bold',
            arrows=True,
            arrowstyle='->',
            arrowsize=20,
            edge_color='gray',
            width=1.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Вместо plt.show() используем:
    plt.draw()
    plt.pause(0.1)  # Даем время на отрисовку
    plt.close()  # Закрываем текущий график

def transitive_reduction(Z: np.ndarray) -> np.ndarray:
    """
    Выполняет транзитивную редукцию ориентированного графа.

    Параметры:
    Z: np.ndarray - матрица смежности (n x n), где Z[i,j]=1 означает, что i доминирует j

    Возвращает:
    np.ndarray - матрица смежности после транзитивной редукции
    """
    n = Z.shape[0]
    # Копируем матрицу, чтобы не изменять исходную
    reduced = Z.copy()

    # Для всех троек вершин (i, j, k)
    for i in range(n):
        for j in range(n):
            if i != j and reduced[i, j] == 1:
                # Проверяем, есть ли путь из i в j через какую-либо вершину k
                for k in range(n):
                    if k != i and k != j:
                        # Если есть путь i -> k и k -> j
                        if reduced[i, k] == 1 and reduced[k, j] == 1:
                            # То ребро i -> j транзитивно и его можно удалить
                            reduced[i, j] = 0
                            break

    return reduced

def transitive_reduction_fast(Z: np.ndarray) -> np.ndarray:
    """
    Более быстрая версия транзитивной редукции с использованием матричного умножения.

    Параметры:
    Z: np.ndarray - матрица смежности (n x n)

    Возвращает:
    np.ndarray - матрица смежности после транзитивной редукции
    """
    n = Z.shape[0]
    reduced = Z.copy()

    Z_squared = np.matmul(Z, Z)

    # Если есть путь длины 2 и есть прямое ребро, удаляем прямое
    for i in range(n):
        for j in range(n):
            if i != j and reduced[i, j] == 1 and Z_squared[i, j] > 0:
                reduced[i, j] = 0

    return reduced

def transitive_closure(Z: np.ndarray) -> np.ndarray:
    """
    Транзитивное замыкание (алгоритм Флойда-Уоршелла)
    """
    n = Z.shape[0]
    closure = Z.copy()

    # Добавляем петли
    for i in range(n):
        closure[i, i] = 1

    # Алгоритм Флойда-Уоршелла для транзитивного замыкания
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if closure[i, k] and closure[k, j]:
                    closure[i, j] = 1

    return closure

# Пример использования с вашими матрицами:
if __name__ == "__main__":


    number = 5
    # Линейный порядок
    Z_linear = PrefMaGen_linear(number)
    []
    visualize_preference_matrix(Z_linear)
    print(Z_linear, f"\n", matrix_to_ordering(Z_linear), "Граф линейного порядка")

    Z_linear_reducto=transitive_reduction(Z_linear)
    visualize_preference_matrix(Z_linear_reducto)
    print(Z_linear_reducto, f"\n", matrix_to_ordering(Z_linear_reducto), "Граф редуцированного линейного порядка")

    Z_linear_reborn=transitive_closure(Z_linear_reducto)
    visualize_preference_matrix(Z_linear_reborn)
    print(Z_linear_reborn, f"\n", matrix_to_ordering(Z_linear_reborn), "Граф восстановленного линейного порядка", np.array_equal(Z_linear_reborn, Z_linear))
    # С уровнями
    Z_level = PrefMaGen_level(number)
    visualize_preference_matrix(Z_level)
    print(Z_level, f"\n", matrix_to_ordering(Z_level), "Граф иерархии")

    Z_level_reducto = transitive_reduction(Z_level)
    visualize_preference_matrix(Z_level_reducto, "Граф редуцированной иерархии")
    print(Z_level_reducto, f"\n", matrix_to_ordering(Z_level_reducto), "Граф редуцированной иерархии")

    Z_level_reborn = transitive_closure(Z_level_reducto)
    visualize_preference_matrix(Z_level_reborn, "Граф восстановленной иерархии")
    print(Z_level_reborn, f"\n", matrix_to_ordering(Z_level_reborn), "Граф восстановленной иерархии", np.array_equal(Z_level_reborn, Z_level))

    # С шумом
    Z_noisy = PrefMaGen_noise(Z_linear, 0.2)
    visualize_preference_matrix(Z_noisy)
    print(Z_noisy, f"\n", matrix_to_ordering(Z_noisy), "Граф с шумом")

    Z_noisy_reducto = transitive_reduction(Z_noisy)
    visualize_preference_matrix(Z_noisy_reducto)
    print(Z_noisy_reducto, f"\n", matrix_to_ordering(Z_noisy_reducto), "Граф с редуцированным шумом")

    Z_noisy_reborn = transitive_closure(Z_noisy_reducto)
    visualize_preference_matrix(Z_noisy_reborn, "Граф восстановленного из шума")
    print(Z_noisy_reborn, f"\n", matrix_to_ordering(Z_noisy_reborn), "Граф восстановленного из шума", np.array_equal(Z_noisy_reborn, Z_noisy))