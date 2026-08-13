import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ ВИЗУАЛИЗАЦИИ - УЛУЧШЕННАЯ ЧИТАЕМОСТЬ
# ============================================================================

OUTPUT_DIR = "../analysis_output/latest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. КАСТОМНЫЕ НАЗВАНИЯ ДЛЯ МЕТОДОВ
# ----------------------------------------------------------------------------
METHOD_NAMES = {
    'Bulatov_matrix': 'Матр. апп.',
    'Bulatov_rank': 'Ранг. апп.',
    'Cohen': 'Hedge',
    'cone_direct': 'Конус пр.',
    'cone_reg': 'Конус рег.',
    'graph_method': 'Граф.'
}

GEN_METHOD_NAMES = {
    'level': 'уровн.',
    'noise': 'шум.'
}

# ----------------------------------------------------------------------------
# 2. ЦВЕТОВАЯ ПАЛИТРА
# ----------------------------------------------------------------------------
VIOLIN_COLORS = {
    'level': '#2E86AB',  # Синий
    'noise': '#A23B72',  # Фиолетовый
}

# ЦВЕТА ДЛЯ ОСТАЛЬНЫХ ГРАФИКОВ (Ч/Б)
GRAY_SCALE = {
    'Bulatov_matrix': '#1a1a1a',
    'Bulatov_rank': '#404040',
    'Cohen': '#666666',
    'cone_direct': '#8c8c8c',
    'cone_reg': '#b3b3b3',
    'graph_method': '#cccccc',
    'level': '#e6e6e6',
    'noise': '#999999',
}

# ----------------------------------------------------------------------------
# 3. МАРКЕРЫ И СТИЛИ
# ----------------------------------------------------------------------------
MARKERS = {
    'Bulatov_matrix': 'o',
    'Bulatov_rank': 's',
    'Cohen': '^',
    'cone_direct': 'v',
    'cone_reg': 'D',
    'graph_method': 'p',
    'level': '*',
    'noise': 'X',
}

LINE_STYLES = {
    'Bulatov_matrix': '-',
    'Bulatov_rank': '--',
    'Cohen': '-.',
    'cone_direct': ':',
    'cone_reg': '-',
    'graph_method': '--',
    'level': '-.',
    'noise': ':',
}

# ----------------------------------------------------------------------------
# 4. НАСТРОЙКИ ЛЕГЕНД
# ----------------------------------------------------------------------------
SIZE_VS_QUALITY_LEGEND = {
    0: {'show': False},
    1: {'show': False},
    2: {'show': True, 'position': 'best', 'ncol': 2},
    3: {'show': False},
    4: {'show': False},
}

SIZE_VS_TIME_LEGEND = {
    'linear': {'show': False},
    'loglog': {'show': True, 'position': 'upper left', 'ncol': 2},
}

VIOLIN_LEGEND = {
    0: {'show': True, 'position': 'lower left', 'ncol': 1},
    1: {'show': True, 'position': 'lower left', 'ncol': 1},
    2: {'show': True, 'position': 'lower right', 'ncol': 1},
    3: {'show': False},
    4: {'show': True, 'position': 'lower right', 'ncol': 1},
}

BOXPLOT_LEGEND = {
    0: {'show': True, 'position': 'upper right', 'ncol': 1},
    1: {'show': False},
    2: {'show': False},
    3: {'show': False},
    4: {'show': True, 'position': 'lower right', 'ncol': 1},
}

GEN_COMPARISON_LEGEND = {
    'default': {
        'honest': {'show': True, 'position': 'upper right', 'ncol': 2},
        'ideal': {'show': False},
    },
    'Bulatov_matrix': {
        'honest': {'show': True, 'position': 'upper left', 'ncol': 1},
        'ideal': {'show': True, 'position': 'lower right', 'ncol': 2},
    },
    'Cohen': {
        'honest': {'show': False},
        'ideal': {'show': True, 'position': 'best', 'ncol': 2},
    },
}

RANKING_LEGEND = {
    'show': True,
    'position': 'lower right',
    'ncol': 1,
}

# ----------------------------------------------------------------------------
# 5. НАСТРОЙКИ ТЕПЛОВОЙ КАРТЫ
# ----------------------------------------------------------------------------
HEATMAP_CONFIG = {
    'show_cbar': True,
    'cbar_label': 'Kendall τ',
    'annot': True,
    'fmt': '.3f',
    'cmap': 'RdYlGn',
    'center': 0.5,
    'vmin': 0,
    'vmax': 1,
    'square': False,
    'linewidths': 0.5,
    'linecolor': 'black',
    'figsize': (5, 5),
    'xtick_rotation': 45,
    'ytick_fontsize': 10,
    'xtick_fontsize': 10,
    'title_fontsize': 14,
    'cbar_shrink': 0.8,
    'cbar_aspect': 20,
}

# ----------------------------------------------------------------------------
# 6. ГЛОБАЛЬНЫЕ НАСТРОЙКИ СТИЛЯ
# ----------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 9

print("=" * 80)
print("📊 АНАЛИЗ ВОССТАНОВЛЕНИЯ ПОРЯДКА")
print("=" * 80)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

df1 = pd.read_csv('../experiment_output/graph_pref_restoration_results.csv')
df2 = pd.read_csv('../experiment_output/latest_pref_restoration_results.csv')
df = pd.concat([df1, df2], ignore_index=True)
df=df2
if 'n_experts' in df.columns and 'm' in df.columns:
    df['size'] = df['m']
elif 'size' not in df.columns:
    df['size'] = 1

df['method_label'] = df['method'].map(METHOD_NAMES).fillna(df['method'])
df['gen_label'] = df['gen_method'].map(GEN_METHOD_NAMES).fillna(df['gen_method'])

hon_metrics = ['hon_my_kendall', 'hon_my_spearman', 'hon_scipy_kendall',
               'hon_scipy_spearman', 'hon_pair_accuracy']
metric_labels = ['Kendall τ', 'Spearman ρ', 'Kendall τ (scipy)', 'Spearman ρ (scipy)', 'Pair Accuracy']


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_method_style(method):
    base_method = method
    for key in METHOD_NAMES.keys():
        if key in method:
            base_method = key
            break
    return {
        'color': GRAY_SCALE.get(base_method, '#666666'),
        'marker': MARKERS.get(base_method, 'o'),
        'linestyle': LINE_STYLES.get(base_method, '-'),
        'label': METHOD_NAMES.get(base_method, method)
    }


def apply_legend(ax, config):
    if config is None:
        return
    if not config.get('show', False):
        return

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    handles = list(unique.values())
    labels = list(unique.keys())

    if not handles:
        return

    position = config.get('position', 'best')
    ncol = config.get('ncol', 1)
    fontsize = config.get('fontsize', 10)
    title = config.get('title', None)
    frameon = config.get('frameon', True)
    fancybox = config.get('fancybox', True)
    shadow = config.get('shadow', True)

    ax.legend(handles, labels,
              loc=position,
              ncol=ncol,
              fontsize=fontsize,
              title=title,
              frameon=frameon,
              fancybox=fancybox,
              shadow=shadow)


# ============================================================================
# 1. ВЛИЯНИЕ РАЗМЕРНОСТИ НА КАЧЕСТВО
# ============================================================================
print("\n📏 1. АНАЛИЗ ВЛИЯНИЯ РАЗМЕРНОСТИ НА КАЧЕСТВО")

if 'size' in df.columns and df['size'].nunique() > 1:

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    methods = df['method'].unique()
    gen_methods = df['gen_method'].unique()

    for idx, (metric, label) in enumerate(zip(hon_metrics[:5], metric_labels[:5])):
        ax = axes[idx]

        for method in methods:
            style = get_method_style(method)
            for gen_method in gen_methods:
                subset = df[(df['method'] == method) & (df['gen_method'] == gen_method)]
                if len(subset) > 0:
                    grouped = subset.groupby('size')[metric].agg(['mean', 'std', 'count']).reset_index()
                    grouped = grouped.sort_values('size')
                    if len(grouped) > 0:
                        ax.plot(grouped['size'], grouped['mean'],
                                color=style['color'],
                                marker=style['marker'],
                                linestyle=style['linestyle'],
                                linewidth=2.5,
                                markersize=8,
                                markeredgecolor='black',
                                markeredgewidth=0.5,
                                label=f"{style['label']} ({GEN_METHOD_NAMES.get(gen_method, gen_method)})")
                        ax.fill_between(grouped['size'],
                                        grouped['mean'] - grouped['std'],
                                        grouped['mean'] + grouped['std'],
                                        color=style['color'],
                                        alpha=0.1)

        ax.set_xlabel('Размерность задачи (m)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)

        config = SIZE_VS_QUALITY_LEGEND.get(idx)
        apply_legend(ax, config)

    if len(axes) > 5:
        fig.delaxes(axes[5])

    plt.suptitle('Зависимость качества восстановления от размерности задачи', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'size_vs_quality.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ size_vs_quality.png")

# ============================================================================
# 2. ВЛИЯНИЕ РАЗМЕРНОСТИ НА ВРЕМЯ
# ============================================================================
print("\n⏱️ 2. АНАЛИЗ ВЛИЯНИЯ РАЗМЕРНОСТИ НА ВРЕМЯ")

if 'size' in df.columns and df['size'].nunique() > 1:

    # 2.1 Линейный масштаб
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in methods:
        style = get_method_style(method)
        for gen_method in gen_methods:
            subset = df[(df['method'] == method) & (df['gen_method'] == gen_method)]
            if len(subset) > 0:
                grouped = subset.groupby('size')['time_seconds'].agg(['mean', 'std']).reset_index()
                grouped = grouped.sort_values('size')
                if len(grouped) > 0:
                    ax.plot(grouped['size'], grouped['mean'],
                            color=style['color'],
                            marker=style['marker'],
                            linestyle=style['linestyle'],
                            linewidth=2.5,
                            markersize=8,
                            markeredgecolor='black',
                            markeredgewidth=0.5,
                            label=f"{style['label']} ({GEN_METHOD_NAMES.get(gen_method, gen_method)})")

    ax.set_xlabel('Размерность задачи (m)', fontsize=12)
    ax.set_ylabel('Время выполнения (сек)', fontsize=12)
    ax.set_title('Зависимость времени от размерности (линейный масштаб)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    apply_legend(ax, SIZE_VS_TIME_LEGEND.get('linear'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'size_vs_time_linear.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ size_vs_time_linear.png")

    # 2.2 Лог-лог масштаб
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in methods:
        style = get_method_style(method)
        for gen_method in gen_methods:
            subset = df[(df['method'] == method) & (df['gen_method'] == gen_method)]
            if len(subset) > 0:
                grouped = subset.groupby('size')['time_seconds'].mean().reset_index()
                grouped = grouped.sort_values('size')
                if len(grouped) > 0 and (grouped['size'] > 0).all() and (grouped['time_seconds'] > 0).all():
                    ax.loglog(grouped['size'], grouped['time_seconds'],
                              color=style['color'],
                              marker=style['marker'],
                              linestyle=style['linestyle'],
                              linewidth=2.5,
                              markersize=8,
                              markeredgecolor='black',
                              markeredgewidth=0.5,
                              label=f"{style['label']} ({GEN_METHOD_NAMES.get(gen_method, gen_method)})")

    ax.set_xlabel('Размерность задачи (логарифм)', fontsize=12)
    ax.set_ylabel('Время выполнения (логарифм, сек)', fontsize=12)
    ax.set_title('Временная сложность (лог-лог масштаб)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    apply_legend(ax, SIZE_VS_TIME_LEGEND.get('loglog'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'size_vs_time_loglog.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ size_vs_time_loglog.png")

# ============================================================================
# 3. СКРИПИЧНЫЕ ГРАФИКИ (ЦВЕТНЫЕ)
# ============================================================================
print("\n📈 3. СКРИПИЧНЫЕ ГРАФИКИ (ЦВЕТНЫЕ)")


def create_colored_violin(data, metric, label, metric_idx):
    """Создает скрипичный график с цветным разделением по методам генерации"""
    fig, ax = plt.subplots(figsize=(14, 7))

    methods = data['method'].unique()
    gen_methods = sorted(data['gen_method'].unique())

    # Определяем порядок методов для красивой группировки
    method_order = ['Bulatov_matrix', 'Bulatov_rank', 'Cohen', 'cone_direct', 'cone_reg', 'graph_method']
    methods = [m for m in method_order if m in methods]

    x_positions = np.arange(len(methods))
    width = 0.8 / len(gen_methods) if len(gen_methods) > 0 else 0.8

    # Собираем данные для каждого метода генерации
    for i, gen in enumerate(gen_methods):
        gen_data = data[data['gen_method'] == gen]
        positions = x_positions + width * (i - (len(gen_methods) - 1) / 2)

        plot_data = []
        for method in methods:
            vals = gen_data[gen_data['method'] == method][metric].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
            else:
                plot_data.append([])

        # Создаем скрипки с цветом
        parts = ax.violinplot(plot_data, positions=positions, widths=width,
                              showmeans=False, showmedians=True, showextrema=False)

        # Применяем цвет
        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
            pc.set_alpha(0.7)

        # Настраиваем медианы
        if parts['cmedians'] is not None:
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2)

    # Настройка осей
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=45, ha='right', fontsize=16,
                       fontweight='bold')
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label} - распределение по методам', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(-0.15, 1.05)

    # Добавляем горизонтальную линию на 0.5 для референса
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Создаем легенду
    legend_elements = []
    for gen in gen_methods:
        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        element = plt.Line2D([0], [0], color=color, marker='s', linestyle='None',
                             markersize=12, markeredgecolor='black',
                             label=GEN_METHOD_NAMES.get(gen, gen))
        legend_elements.append(element)

    # Применяем настройки легенды
    config = VIOLIN_LEGEND.get(metric_idx)
    if config and config.get('show', False):
        position = config.get('position', 'best')
        ncol = config.get('ncol', 1)
        ax.legend(handles=legend_elements,
                  labels=[GEN_METHOD_NAMES.get(gen, gen) for gen in gen_methods],
                  loc=position,
                  ncol=ncol,
                  fontsize=11,
                  title='Метод генерации',
                  title_fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  facecolor='white',
                  edgecolor='black')

    plt.tight_layout()
    return fig


# Создаем скрипичные графики для каждой метрики
for idx, (metric, label) in enumerate(zip(hon_metrics[:5], metric_labels[:5])):
    fig = create_colored_violin(df, metric, label, idx)
    fig.savefig(os.path.join(OUTPUT_DIR, f'violin_colored_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ violin_colored_{metric}.png")

# ============================================================================
# 3.1 БОКСПЛОТЫ (ЦВЕТНЫЕ)
# ============================================================================
print("\n📊 3.1 БОКСПЛОТЫ (ЦВЕТНЫЕ)")


def create_colored_boxplot(data, metric, label, metric_idx):
    """Создает боксплот с цветным разделением по методам генерации"""
    fig, ax = plt.subplots(figsize=(14, 7))

    methods = data['method'].unique()
    gen_methods = sorted(data['gen_method'].unique())

    # Определяем порядок методов для красивой группировки
    method_order = ['Bulatov_matrix', 'Bulatov_rank', 'Cohen', 'cone_direct', 'cone_reg', 'graph_method']
    methods = [m for m in method_order if m in methods]

    x_positions = np.arange(len(methods))
    width = 0.8 / len(gen_methods) if len(gen_methods) > 0 else 0.8

    # Собираем данные для каждого метода генерации
    for i, gen in enumerate(gen_methods):
        gen_data = data[data['gen_method'] == gen]
        positions = x_positions + width * (i - (len(gen_methods) - 1) / 2)

        plot_data = []
        for method in methods:
            vals = gen_data[gen_data['method'] == method][metric].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
            else:
                plot_data.append([])

        # Создаем боксплоты с цветом
        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        bp = ax.boxplot(plot_data, positions=positions, widths=width * 0.8,
                        patch_artist=True,
                        showmeans=False,
                        meanline=False,
                        showbox=True,
                        showcaps=True,
                        showfliers=True,
                        whiskerprops={'color': 'black', 'linewidth': 1.5},
                        capprops={'color': 'black', 'linewidth': 1.5},
                        medianprops={'color': 'black', 'linewidth': 2},
                        flierprops={'marker': 'o', 'markerfacecolor': color,
                                    'markeredgecolor': 'black', 'markersize': 6})

        # Закрашиваем боксплоты
        for box in bp['boxes']:
            box.set_facecolor(color)
            box.set_alpha(0.7)
            box.set_edgecolor('black')
            box.set_linewidth(1.5)

    # Настройка осей
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=45, ha='right')
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label} - распределение по методам (Boxplot)', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(-0.15, 1.05)

    # Добавляем горизонтальную линию на 0.5 для референса
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Создаем легенду
    legend_elements = []
    for gen in gen_methods:
        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        element = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black',
                                alpha=0.7, label=GEN_METHOD_NAMES.get(gen, gen))
        legend_elements.append(element)

    # Применяем настройки легенды
    config = BOXPLOT_LEGEND.get(metric_idx)
    if config and config.get('show', False):
        position = config.get('position', 'best')
        ncol = config.get('ncol', 1)
        ax.legend(handles=legend_elements,
                  labels=[GEN_METHOD_NAMES.get(gen, gen) for gen in gen_methods],
                  loc=position,
                  ncol=ncol,
                  fontsize=11,
                  title='Метод генерации',
                  title_fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  facecolor='white',
                  edgecolor='black')

    plt.tight_layout()
    return fig


# Создаем боксплоты для каждой метрики
for idx, (metric, label) in enumerate(zip(hon_metrics[:5], metric_labels[:5])):
    fig = create_colored_boxplot(df, metric, label, idx)
    fig.savefig(os.path.join(OUTPUT_DIR, f'boxplot_colored_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ boxplot_colored_{metric}.png")

# ============================================================================
# 3.2 СОВМЕЩЕННЫЕ ГРАФИКИ (VIOLIN + BOXPLOT)
# ============================================================================
print("\n📊 3.2 СОВМЕЩЕННЫЕ ГРАФИКИ (VIOLIN + BOXPLOT)")


def create_violin_boxplot_combined(data, metric, label, metric_idx):
    """Создает совмещенный график: скрипка + боксплот внутри"""
    fig, ax = plt.subplots(figsize=(14, 7))

    methods = data['method'].unique()
    gen_methods = sorted(data['gen_method'].unique())

    # Определяем порядок методов для красивой группировки
    method_order = ['Bulatov_matrix', 'Bulatov_rank', 'Cohen', 'cone_direct', 'cone_reg', 'graph_method']
    methods = [m for m in method_order if m in methods]

    x_positions = np.arange(len(methods))
    width = 0.8 / len(gen_methods) if len(gen_methods) > 0 else 0.8

    # Собираем данные для каждого метода генерации
    for i, gen in enumerate(gen_methods):
        gen_data = data[data['gen_method'] == gen]
        positions = x_positions + width * (i - (len(gen_methods) - 1) / 2)

        plot_data = []
        for method in methods:
            vals = gen_data[gen_data['method'] == method][metric].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
            else:
                plot_data.append([])

        # Создаем скрипки
        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        parts = ax.violinplot(plot_data, positions=positions, widths=width,
                              showmeans=False, showmedians=False, showextrema=False)

        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
            pc.set_alpha(0.5)

        # Добавляем боксплот поверх скрипки
        bp = ax.boxplot(plot_data, positions=positions, widths=width * 0.4,
                        patch_artist=True,
                        showmeans=False,
                        meanline=False,
                        showbox=True,
                        showcaps=True,
                        showfliers=False,
                        whiskerprops={'color': 'black', 'linewidth': 1.5},
                        capprops={'color': 'black', 'linewidth': 1.5},
                        medianprops={'color': 'white', 'linewidth': 2})

        for box in bp['boxes']:
            box.set_facecolor('white')
            box.set_alpha(0.7)
            box.set_edgecolor('black')
            box.set_linewidth(1.5)

    # Настройка осей
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=45, ha='right')
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label} - распределение по методам (Violin + Boxplot)', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(-0.15, 1.05)

    # Добавляем горизонтальную линию на 0.5 для референса
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Создаем легенду
    legend_elements = []
    for gen in gen_methods:
        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        element = plt.Line2D([0], [0], color=color, marker='s', linestyle='None',
                             markersize=12, markeredgecolor='black',
                             label=GEN_METHOD_NAMES.get(gen, gen))
        legend_elements.append(element)

    # Применяем настройки легенды
    config = VIOLIN_LEGEND.get(metric_idx)
    if config and config.get('show', False):
        position = config.get('position', 'best')
        ncol = config.get('ncol', 1)
        ax.legend(handles=legend_elements,
                  labels=[GEN_METHOD_NAMES.get(gen, gen) for gen in gen_methods],
                  loc=position,
                  ncol=ncol,
                  fontsize=11,
                  title='Метод генерации',
                  title_fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  facecolor='white',
                  edgecolor='black')

    plt.tight_layout()
    return fig


# Создаем совмещенные графики для каждой метрики
for idx, (metric, label) in enumerate(zip(hon_metrics[:5], metric_labels[:5])):
    fig = create_violin_boxplot_combined(df, metric, label, idx)
    fig.savefig(os.path.join(OUTPUT_DIR, f'violin_boxplot_combined_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ violin_boxplot_combined_{metric}.png")

# ============================================================================
# 4. ТЕПЛОВАЯ КАРТА
# ============================================================================
print("\n🔥 4. ТЕПЛОВАЯ КАРТА")


def create_heatmap(data, metric='hon_my_kendall', title='HONEST Kendall τ для каждой размерности'):
    pivot_data = []

    for method in data['method'].unique():
        for gen_method in data['gen_method'].unique():
            subset = data[(data['method'] == method) & (data['gen_method'] == gen_method)]

            if len(subset) > 0:
                row_name = f"{METHOD_NAMES.get(method, method)} ({GEN_METHOD_NAMES.get(gen_method, gen_method)})"

                for size_val in subset['size'].unique():
                    val = subset[subset['size'] == size_val][metric].mean()
                    if pd.notna(val):
                        pivot_data.append({
                            'Метод + Генерация': row_name,
                            'Размерность': size_val,
                            'Значение': val
                        })

    if not pivot_data:
        print("⚠️ Нет данных для тепловой карты")
        return None

    pivot_df = pd.DataFrame(pivot_data)
    pivot_table = pivot_df.pivot_table(
        values='Значение',
        index='Метод + Генерация',
        columns='Размерность',
        aggfunc='mean'
    )

    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
    row_order = pivot_table.mean(axis=1).sort_values(ascending=False).index
    pivot_table = pivot_table.loc[row_order]

    config = HEATMAP_CONFIG

    fig, ax = plt.subplots(figsize=config.get('figsize', (14, 10)))

    heatmap = sns.heatmap(
        pivot_table,
        annot=config.get('annot', True),
        fmt=config.get('fmt', '.3f'),
        cmap=config.get('cmap', 'RdYlGn'),
        center=config.get('center', 0.5),
        vmin=config.get('vmin', 0),
        vmax=config.get('vmax', 1),
        ax=ax,
        cbar=config.get('show_cbar', True),
        cbar_kws={
            'label': config.get('cbar_label', metric),
            'shrink': config.get('cbar_shrink', 0.8),
            'aspect': config.get('cbar_aspect', 20),
        },
        square=config.get('square', True),
        linewidths=config.get('linewidths', 0.5),
        linecolor=config.get('linecolor', 'black'),
    )

    ax.set_title(title, fontsize=config.get('title_fontsize', 14), fontweight='bold', pad=20)
    ax.set_xlabel('Размерность задачи (m)', fontsize=12)
    ax.set_ylabel('Метод восстановления (метод генерации)', fontsize=12)

    plt.setp(ax.get_xticklabels(),
             rotation=config.get('xtick_rotation', 45),
             ha='right',
             fontsize=config.get('xtick_fontsize', 10))
    plt.setp(ax.get_yticklabels(),
             fontsize=config.get('ytick_fontsize', 10))

    plt.tight_layout()
    return fig, pivot_table


heatmap_metrics = [
    ('hon_my_kendall', 'HONEST Kendall τ для каждой размерности'),
    ('hon_my_spearman', 'HONEST Spearman ρ для каждой размерности'),
    ('hon_pair_accuracy', 'HONEST Pair Accuracy для каждой размерности'),
    ('ideal_my_kendall', 'IDEAL Kendall τ для каждой размерности'),
]

for metric, title in heatmap_metrics:
    result = create_heatmap(df, metric, title)
    if result is not None:
        fig, pivot_table = result
        filename = f'heatmap_{metric}.png'
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ {filename}")

# ============================================================================
# 5. СРАВНЕНИЕ МЕТОДОВ ГЕНЕРАЦИИ
# ============================================================================
print("\n📊 5. СРАВНЕНИЕ МЕТОДОВ ГЕНЕРАЦИИ")

for method in df['method'].unique():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    data_method = df[df['method'] == method]
    gen_methods = data_method['gen_method'].unique()
    metrics_to_plot = ['hon_my_kendall', 'hon_my_spearman', 'hon_pair_accuracy']
    labels_to_plot = ['Kendall τ', 'Spearman ρ', 'Pair Accuracy']

    x_pos = np.arange(len(metrics_to_plot))
    width = 0.8 / len(gen_methods) if len(gen_methods) > 0 else 0.8

    # HONEST
    for i, gen in enumerate(gen_methods):
        gen_data = data_method[data_method['gen_method'] == gen]
        means = [gen_data[m].mean() for m in metrics_to_plot]
        stds = [gen_data[m].std() for m in metrics_to_plot]
        offset = width * (i - (len(gen_methods) - 1) / 2)

        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        axes[0].bar(x_pos + offset, means, width,
                    color=color,
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.7,
                    label=GEN_METHOD_NAMES.get(gen, gen))
        axes[0].errorbar(x_pos + offset, means, yerr=stds,
                         fmt='none', color='black', capsize=5, capthick=2)

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels_to_plot, rotation=45)
    axes[0].set_ylabel('Значение метрики', fontsize=12)
    axes[0].set_title(f'HONEST метрики', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].set_ylim(-0.15, 1.05)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

    method_config = GEN_COMPARISON_LEGEND.get(method, GEN_COMPARISON_LEGEND['default'])
    apply_legend(axes[0], method_config.get('honest'))

    # IDEAL
    for i, gen in enumerate(gen_methods):
        gen_data = data_method[data_method['gen_method'] == gen]
        means = [gen_data[f'ideal_{m[4:]}'].mean() for m in metrics_to_plot]
        stds = [gen_data[f'ideal_{m[4:]}'].std() for m in metrics_to_plot]
        offset = width * (i - (len(gen_methods) - 1) / 2)

        color = VIOLIN_COLORS.get(gen, '#2E86AB')
        axes[1].bar(x_pos + offset, means, width,
                    color=color,
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.7,
                    label=GEN_METHOD_NAMES.get(gen, gen))
        axes[1].errorbar(x_pos + offset, means, yerr=stds,
                         fmt='none', color='black', capsize=5, capthick=2)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels_to_plot, rotation=45)
    axes[1].set_ylabel('Значение метрики', fontsize=12)
    axes[1].set_title(f'IDEAL метрики', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].set_ylim(-0.15, 1.05)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

    apply_legend(axes[1], method_config.get('ideal'))

    plt.suptitle(f'Сравнение методов генерации для {METHOD_NAMES.get(method, method)}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gen_comparison_{method}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ gen_comparison_{method}.png")

# ============================================================================
# 6. РЕЙТИНГ МЕТОДОВ
# ============================================================================
print("\n📊 6. РЕЙТИНГ МЕТОДОВ")

fig, ax = plt.subplots(figsize=(14, 8))

metrics_for_ranking = ['hon_my_kendall', 'hon_my_spearman', 'hon_pair_accuracy']

method_scores = {}
for method in df['method'].unique():
    method_data = df[df['method'] == method]
    scores = [method_data[metric].mean() for metric in metrics_for_ranking]
    method_scores[method] = np.mean(scores)

sorted_methods = sorted(method_scores.keys(), key=lambda x: method_scores[x], reverse=True)
positions = np.arange(len(sorted_methods))

for i, method in enumerate(sorted_methods):
    style = get_method_style(method)
    method_data = df[df['method'] == method]
    mean_score = method_scores[method]
    std_score = method_data[metrics_for_ranking].mean(axis=1).std()

    ax.errorbar(i, mean_score, yerr=std_score,
                fmt=style['marker'],
                color=style['color'],
                markersize=14,
                markeredgecolor='black',
                markeredgewidth=1.5,
                capsize=5,
                capthick=2,
                elinewidth=2,
                label=style['label'])

ax.set_xticks(positions)
ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in sorted_methods], rotation=45, ha='right')
ax.set_ylabel('Средний балл по метрикам', fontsize=12)
ax.set_title('Рейтинг методов восстановления\n(среднее по Kendall + Spearman + Pair Accuracy)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(-0.15, 1.05)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

apply_legend(ax, RANKING_LEGEND)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'method_ranking.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ method_ranking.png")

# ============================================================================
# 7. ИТОГИ
# ============================================================================
print("\n" + "=" * 80)
print("✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
print("=" * 80)
print(f"\n📁 Все результаты в: {OUTPUT_DIR}")
print("\n🎨 Созданные графики:")
print("   📈 Скрипичные графики (violin_colored_*.png)")
print("   📊 Боксплоты (boxplot_colored_*.png)")
print("   🎻 Совмещенные графики (violin_boxplot_combined_*.png)")
print("\nЦветовая схема:")
print("   - Синий (#2E86AB) - метод генерации 'level'")
print("   - Фиолетовый (#A23B72) - метод генерации 'noise'")
print("=" * 80)