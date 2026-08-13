from pathlib import Path


def combine_py_files(directories=['.'], output='combined.txt'):
    """
    Объединяет все .py файлы из указанных директорий в один файл.

    Args:
        directories: список путей к папкам (по умолчанию только текущая папка)
        output: имя выходного файла
    """
    with open(output, 'w', encoding='utf-8') as outfile:
        for directory in directories:
            dir_path = Path(directory)

            # Проверяем, существует ли папка
            if not dir_path.exists():
                outfile.write(f"\n{'!' * 60}\n")
                outfile.write(f"ПРЕДУПРЕЖДЕНИЕ: Папка '{directory}' не найдена!\n")
                outfile.write(f"{'!' * 60}\n\n")
                continue

            # Записываем заголовок папки
            outfile.write(f"\n{'#' * 60}\n")
            outfile.write(f"## ПАПКА: {dir_path.absolute()}\n")
            outfile.write(f"{'#' * 60}\n\n")

            # Находим все .py файлы в текущей папке (не рекурсивно)
            py_files = sorted(dir_path.glob('*.py'))

            if not py_files:
                outfile.write(f"(Нет .py файлов в папке {directory})\n\n")
                continue

            # Обрабатываем каждый .py файл
            for py_file in py_files:
                outfile.write(f"\n{'=' * 60}\n")
                # Исправление: просто пишем имя файла или относительный путь
                try:
                    # Пытаемся сделать путь относительно текущей директории
                    rel_path = py_file.relative_to(Path.cwd())
                    outfile.write(f"Файл: {rel_path}\n")
                except ValueError:
                    # Если не получается, пишем абсолютный путь
                    outfile.write(f"Файл: {py_file}\n")
                outfile.write(f"{'=' * 60}\n\n")

                try:
                    with open(py_file, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        if not content.endswith('\n'):
                            outfile.write('\n')
                    outfile.write('\n')
                except Exception as e:
                    outfile.write(f"Ошибка чтения: {e}\n\n")


def combine_py_files_recursive(directories=['.'], output='combined.txt'):
    """
    Рекурсивно объединяет все .py файлы из указанных директорий.
    Включает файлы из всех подпапок.
    """
    with open(output, 'w', encoding='utf-8') as outfile:
        for directory in directories:
            dir_path = Path(directory)

            if not dir_path.exists():
                outfile.write(f"\nПРЕДУПРЕЖДЕНИЕ: Папка '{directory}' не найдена!\n\n")
                continue

            # Рекурсивный поиск всех .py файлов
            py_files = sorted(dir_path.rglob('*.py'))

            outfile.write(f"\n{'#' * 60}\n")
            outfile.write(f"## ПАПКА: {dir_path.absolute()}\n")
            outfile.write(f"## Всего файлов: {len(py_files)}\n")
            outfile.write(f"{'#' * 60}\n\n")

            for py_file in py_files:
                outfile.write(f"\n{'=' * 60}\n")
                # Исправление: безопасное получение относительного пути
                try:
                    rel_path = py_file.relative_to(Path.cwd())
                    outfile.write(f"Файл: {rel_path}\n")
                except ValueError:
                    outfile.write(f"Файл: {py_file}\n")
                outfile.write(f"{'=' * 60}\n\n")

                try:
                    with open(py_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n\n')
                except Exception as e:
                    outfile.write(f"Ошибка чтения: {e}\n\n")


def combine_py_files_simple(directories=['.'], output='combined.txt'):
    """
    Упрощённая версия - просто пишет полные пути к файлам.
    """
    with open(output, 'w', encoding='utf-8') as outfile:
        for directory in directories:
            dir_path = Path(directory)

            if not dir_path.exists():
                outfile.write(f"\n[ПРЕДУПРЕЖДЕНИЕ] Папка '{directory}' не найдена!\n\n")
                continue

            outfile.write(f"\n{'#' * 60}\n")
            outfile.write(f"## ПАПКА: {dir_path.absolute()}\n")
            outfile.write(f"{'#' * 60}\n\n")

            # Рекурсивный поиск всех .py файлов
            py_files = sorted(dir_path.rglob('*.py'))

            for py_file in py_files:
                outfile.write(f"\n{'=' * 60}\n")
                outfile.write(f"Файл: {py_file}\n")  # Просто пишем полный путь
                outfile.write(f"{'=' * 60}\n\n")

                try:
                    with open(py_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n\n')
                except Exception as e:
                    outfile.write(f"Ошибка чтения: {e}\n\n")


if __name__ == '__main__':
    # Указываем папки (можно использовать относительные пути)
    папки = [
        'utility',
        'methods',
        'experiments',
        'archive'
    ]

    print("Выберите режим работы:")
    print("1 - Только указанные папки (без вложенных)")
    print("2 - Рекурсивно (включая все подпапки)")
    print("3 - Упрощённый (рекурсивно, с полными путями)")

    choice = input("Ваш выбор (1/2/3): ").strip()

    if choice == '2':
        combine_py_files_recursive(directories=папки, output='combined_all.txt')
        print("Готово! Результат в combined_all.txt (с вложенными папками)")
    elif choice == '3':
        combine_py_files_simple(directories=папки, output='combined_simple.txt')
        print("Готово! Результат в combined_simple.txt (упрощённая версия)")
    else:
        combine_py_files(directories=папки, output='combined.txt')
        print("Готово! Результат в combined.txt (только указанные папки)")