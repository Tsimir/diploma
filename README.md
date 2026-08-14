# SRW — Восстановление предпочтений

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Описание

Данный проект — дипломная работа по теме **"Методы восстановления предпочтений"**. 
В рамках работы реализованы и сравниваются следующие алгоритмы:

- **Naive SVD** — базовый метод на основе сингулярного разложения
- **Greedy Order-by-Preference (OBP)** — жадный алгоритм обучения ранжированию
- **Conical Methods** — конусные методы (обычный и регуляризованный)

Цель — сравнение эффективности методов на синтетических данных и выявление оптимального подхода.

## 🏗️ Структура проекта
SRW/  
├── methods/                          # Реализации алгоритмов  
│   ├── agregate_graph.py  
│   ├── naive_svd.py  
│   ├── greedy_obp.py  
│   └── conical_methods.py  
├── archive/                      # Неиспользуемые вспомогательные скрипты  
│   ├── agree.py  
│   ├── forreview.py  
│   ├── semkin_kefs.py   
│   └── graph.py  
└── utility/                          # Вспомогательные функции  
│   ├── comparator.py                 # Быстрое сравнение  
│   ├── generator.py                  # Генерация матриц и метрики качества  
│   └── saver.py                      # Сохранение матриц  



## ⚙️ Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Tsimir/SRW.git
cd SRW
```
### 2. Установить зависимости

```bash
pip install -r requirements.txt
```