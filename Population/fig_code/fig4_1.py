import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import re
import matplotlib.patches as mpatches

# 全局字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


def format_pollutant_label(name):
    """格式化标签：将 PM2.5_O3 转换为 PM_{2.5}+O_3 的 LaTeX 格式"""
    if name == 'clean': return 'Clean'
    if name == 'Others': return 'Others'
    name = name.replace('ONLY_', '').replace('ONLY+', '')
    # 替换连接符
    name_display = name.replace('_', '+')
    parts = name_display.split('+')
    formatted = []
    for p in parts:
        # 正则匹配化学式，如 PM2.5 -> PM_{2.5}, O3 -> O_3
        match = re.match(r'^([A-Za-z]+)([\d.]+)?$', p)
        if match:
            letters, numbers = match.groups()
            formatted.append(f"{letters}$_{{{numbers}}}$" if numbers else letters)
        else:
            formatted.append(p)
    return '+'.join(formatted)


def get_composition_data(df, target_cols, top_n=3):
    valid_cols = [c for c in target_cols if c in df.columns]
    if not valid_cols: return None
    # 计算总和
    total = df[valid_cols].sum(axis=1)
    total = total.replace(0, 1)  # 避免除以0
    # 找 Top N
    col_sums = df[valid_cols].sum().sort_values(ascending=False)
    top_cols = col_sums.head(top_n).index.tolist()
    # 计算百分比
    df_pct = pd.DataFrame(index=df.index)
    for col in top_cols:
        df_pct[col] = df[col].div(total)
        # 计算 Others
    others_cols = [c for c in valid_cols if c not in top_cols]
    if others_cols:
        df_pct['Others'] = df[others_cols].sum(axis=1).div(total)
    else:
        df_pct['Others'] = 0.0
    return df_pct.fillna(0)


# ================= 2. 读取数据 (请确保路径正确) =================
df_extended = pd.read_csv("./data/bar_6.csv")
years = np.arange(2003, 2026)
df_extended.set_index('year', inplace=True)
cols_single = ['ONLY_PM2.5', 'ONLY_PM10', 'ONLY_CO', 'ONLY_NO2', 'ONLY_SO2', 'ONLY_O3']
all_two_combinations = ['PM2.5_PM10', 'PM2.5_CO', 'PM2.5_NO2', 'PM2.5_SO2', 'PM2.5_O3', 'PM10_CO', 'PM10_NO2',
                        'PM10_SO2', 'PM10_O3', 'CO_NO2', 'CO_SO2', 'CO_O3', 'NO2_SO2', 'NO2_O3', 'SO2_O3']
all_three_combinations = ['PM2.5_PM10_CO', 'PM2.5_PM10_NO2', 'PM2.5_PM10_SO2', 'PM2.5_PM10_O3', 'PM2.5_CO_NO2',
                          'PM2.5_CO_SO2', 'PM2.5_CO_O3', 'PM2.5_NO2_SO2', 'PM2.5_NO2_O3', 'PM2.5_SO2_O3', 'PM10_CO_NO2',
                          'PM10_CO_SO2', 'PM10_CO_O3', 'PM10_NO2_SO2', 'PM10_NO2_O3', 'PM10_SO2_O3', 'CO_NO2_SO2',
                          'CO_NO2_O3', 'CO_SO2_O3', 'NO2_SO2_O3']
all_four_combinations = ['PM2.5_PM10_CO_NO2', 'PM2.5_PM10_CO_SO2', 'PM2.5_PM10_CO_O3', 'PM2.5_PM10_NO2_SO2',
                         'PM2.5_PM10_NO2_O3', 'PM2.5_PM10_SO2_O3', 'PM2.5_CO_NO2_SO2', 'PM2.5_CO_NO2_O3',
                         'PM2.5_CO_SO2_O3', 'PM2.5_NO2_SO2_O3', 'PM10_CO_NO2_SO2', 'PM10_CO_NO2_O3', 'PM10_CO_SO2_O3',
                         'PM10_NO2_SO2_O3', 'CO_NO2_SO2_O3']
all_five_combinations = ['PM2.5_PM10_CO_NO2_SO2', 'PM2.5_PM10_CO_NO2_O3', 'PM2.5_PM10_CO_SO2_O3',
                         'PM2.5_PM10_NO2_SO2_O3', 'PM2.5_CO_NO2_SO2_O3', 'PM10_CO_NO2_SO2_O3']


def safe_sum(cols):
    valid = [c for c in cols if c in df_extended.columns]
    return df_extended[valid].sum(axis=1).values


data_main = {
    'Clean': df_extended.get('clean_days', np.zeros(len(years))),
    'Single Pollutant': safe_sum(cols_single),
    'Two Pollutants': safe_sum(all_two_combinations),
    'Three Pollutants': safe_sum(all_three_combinations),
    'Four Pollutants': safe_sum(all_four_combinations),
    'Five Pollutants': safe_sum(all_five_combinations),
}
suffix_map = {
    'Single Pollutant': '(1)',
    'Two Pollutants': '(2)',
    'Three Pollutants': '(3)',
    'Four Pollutants': '(4)',
    'Five Pollutants': '(5)'
}
# ================= 3. 颜色定义 =================
# A. 主图颜色
COLORS_MAIN = {
    'Clean': '#F7F7F7',
    'Single Pollutant': '#9CB2D5',
    'Two Pollutants': '#385286',
    'Three Pollutants': '#E6C7C0',
    'Four Pollutants': '#F4E099',
    'Five Pollutants': '#B68366'
}
PALETTES = {
    # (b) 单一污染物: 蓝色系 (Dark -> Light)
    'Single Pollutant': [
        '#7A96BE',  # Rank 1: 加深的灰蓝 (突出主体)
        '#9CB2D5',  # Rank 2: 您指定的主色
        '#B5C5E0',  # Rank 3: 变浅
        '#CED9EB'
    ],

    # (c) 双重污染物: 金/黄色系 (Dark -> Light)
    # 注意：黄色系最深色建议偏褐/金，否则看不清
    'Two Pollutants': [
        '#385286',  # Rank 1: 您指定的主色 (最深)
        '#5F7396',  # Rank 2: 中蓝
        '#8795A7'  # Rank 3: 浅蓝灰
    ],
    # (d) 三重污染物: 橙色系 (Dark -> Light)
    'Three Pollutants': [
        '#C49A92',  # Rank 1: 深藕荷/干玫瑰色
        '#ECD5D0',  # Rank 2: 您指定的主色
        '#F9F1F0'  # Rank 3: 浅粉
    ],

    # (e) 四重污染物: 红色系 (Dark -> Light)
    'Four Pollutants': [
        '#D4AF37',  # Rank 1: 金属金 (确保看不清黄色时的替代)
        '#F4E099',  # Rank 2: 您指定的主色
        '#F7E8B3'  # Rank 3: 奶黄
    ],

    # (f) 五重污染物: 紫色系 (Dark -> Light)
    'Five Pollutants': [
        # '#8C5E42',  # Rank 1: 深栗色 (加深一点以示严重)
        '#B68366',  # Rank 2: 您指定的主色
        '#C9A28C'  # Rank 3: 浅褐
    ]
}
'''
COLORS_MAIN = {
    'Clean':            '#F7F7F7', 
    'Single Pollutant': '#68A0BF',  
    'Two Pollutants':   '#FEDF88', 
    'Three Pollutants': '#FD8D3C', 
    'Four Pollutants':  '#DF4F46',  
    'Five Pollutants':  '#6A6CB4',  
}
# B. 子图详细成分颜色
PALETTES = {
    # (b) 单一污染物: 蓝色系 (Dark -> Light)
    'Single Pollutant': [
        '#08306B', # Rank 1 (最深蓝)
        '#2171B5', # Rank 2
        '#4292C6', # Rank 3
        '#6BAED6', # Rank 4
        '#9ECAE1', # Rank 5
        '#DEEBF7'  # Rank 6 (备用)
    ],

    # (c) 双重污染物: 金/黄色系 (Dark -> Light)
    # 注意：黄色系最深色建议偏褐/金，否则看不清
    'Two Pollutants': [
        '#8C6D31', # Rank 1 (深褐金)
        '#BD9E39', # Rank 2 (暗金)
        '#E7BA52', # Rank 3 (纯金)
        '#E7CB94', # Rank 4 (浅金)
        '#FDD0A2', # Rank 5
        '#FFF7BC'  # Rank 6
    ],

    # (d) 三重污染物: 橙色系 (Dark -> Light)
    'Three Pollutants': [
        '#7F2704', # Rank 1 (深焦褐)
        '#A63603', # Rank 2 (砖红)
        '#D94801', # Rank 3 (深橙)
        '#F16913', # Rank 4 (鲜橙)
        '#FD8D3C', # Rank 5
        '#FDD0A2'  # Rank 6
    ],

    # (e) 四重污染物: 红色系 (Dark -> Light)
    'Four Pollutants': [
        '#67000D', # Rank 1 (黑红)
        '#A50F15', # Rank 2 (深红)
        '#CB181D', # Rank 3
        '#EF3B2C', # Rank 4
        '#FB6A4A', # Rank 5
        '#FC9272'  # Rank 6
    ],

    # (f) 五重污染物: 紫色系 (Dark -> Light)
    'Five Pollutants': [
        '#8C8BC5', # Rank 3
        '#ACAAD5', # Rank 3
        '#CDCCE6', # Rank 4
        '#EEEEF7', # Rank 5
    ]
}
'''
# Others 的专用颜色 (固定为浅灰)
COLOR_OTHERS = '#E0E0E0'


# 映射字典


def get_sub_color(col_name, main_category):
    """根据列名和主分类，自动获取对应色系的颜色"""
    # 1. 转换列名: PM2.5_O3 -> PM2.5+O3
    key_plus = col_name.replace('_', '+')

    # 2. 获取对应色系的字典
    target_dict = SUB_COLOR_MAPS.get(main_category, {})
    # 3. 查找颜色 (如果在字典里就用，不在就用 Others，还不在就用默认灰)
    if key_plus in target_dict:
        return target_dict[key_plus]
    else:
        # 如果是 Others 列，或者未定义的组合
        return target_dict.get('Others', '#D1D1D1')


# ================= 4. 绘图主逻辑 =================

fig = plt.figure(figsize=(18, 8))  # 宽画布，适合左右分栏
gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.08,
                             left=0.05, right=0.98, top=0.92, bottom=0.08)
# -------------------------------------------------------------------------
# Part 1: 左侧大图
# -------------------------------------------------------------------------
ax_main = fig.add_subplot(gs_outer[0])
bottom_val = np.zeros(len(years))
# 堆叠顺序：从 Clean 到 Five
stack_order = ['Clean', 'Single Pollutant', 'Two Pollutants', 'Three Pollutants', 'Four Pollutants', 'Five Pollutants']
# 绘制堆叠柱状图
main_bars = []  # 用于后面提取图例
for label in stack_order:
    data = data_main[label]
    data = np.maximum(data, 0)
    bar = ax_main.bar(years, data, bottom=bottom_val, label=label,
                      color=COLORS_MAIN[label], width=0.7,
                      edgecolor="black", linewidth=0.7, alpha=0.8)
    bottom_val += data
    main_bars.append(bar)
# 装饰左侧图
ax_main.set_title('(a) Trend of Pollution Complexity Levels', loc='left', fontweight='bold', fontsize=14)
ax_main.set_ylabel('Compound Unclean Air Days', fontweight='bold', fontsize=12)
ax_main.set_xlim(2002.5, 2025.5)
ax_main.set_ylim(0, 370)
ax_main.set_yticks([0, 100, 200, 300])
ax_main.set_xticks([2003, 2008, 2013, 2018, 2025])
ax_main.tick_params(axis='both', which='both', length=0)
# 去除多余边框
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)

# -------------------------------------------------------------------------
# Part 2: 右侧小图网格 (Panel B-F + Legend)
# -------------------------------------------------------------------------
# 创建右侧内部网格：2行3列
gs_right = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_outer[1],
                                            wspace=0.18, hspace=0.4)

# 定义右侧 5 个子图的配置
# 顺序：Single, Two, Three (第一行); Four, Five (第二行)
subplots_config = [
    {'title': '(b) Single Pollutant', 'cols': cols_single, 'cat': 'Single Pollutant', 'pos': (0, 0), 'top_n': 4},
    {'title': '(c) Two Pollutants', 'cols': all_two_combinations, 'cat': 'Two Pollutants', 'pos': (0, 1), 'top_n': 3},
    {'title': '(d) Three Pollutants', 'cols': all_three_combinations, 'cat': 'Three Pollutants', 'pos': (0, 2),
     'top_n': 3},
    {'title': '(e) Four Pollutants', 'cols': all_four_combinations, 'cat': 'Four Pollutants', 'pos': (1, 1),
     'top_n': 2},
    {'title': '(f) Five Pollutants', 'cols': all_five_combinations, 'cat': 'Five Pollutants', 'pos': (1, 2),
     'top_n': 1},
]
for cfg in subplots_config:
    row, col = cfg['pos']
    ax = fig.add_subplot(gs_right[row, col])

    # 1. 获取数据
    # get_composition_data 内部已经按照总量从大到小排序了 top_cols
    df_plot = get_composition_data(df_extended, cfg['cols'], top_n=cfg['top_n'])

    if df_plot is not None:
        bottom = np.zeros(len(df_plot))

        # 分离 columns: 确保 'Others' 放在最后绘制（或者最上层）
        # data cols 已经是排好序的 Top N
        data_cols = [c for c in df_plot.columns if c != 'Others']
        has_others = 'Others' in df_plot.columns and df_plot['Others'].sum() > 0

        # 准备绘图列表：先画主要成分，最后画 Others
        cols_to_plot = data_cols + (['Others'] if has_others else [])

        # 获取当前类别的色板
        current_palette = PALETTES[cfg['cat']]

        # 2. 循环绘制每一列
        for idx, col_name in enumerate(cols_to_plot):
            label = format_pollutant_label(col_name)

            # --- [核心修改: 动态颜色分配] ---
            if col_name == 'Others':
                color = COLOR_OTHERS  # Others 固定用灰色
            else:
                # 根据 idx (排名) 从色板中取色
                # idx=0 是第一名(用最深色), idx=1 是第二名...
                # 防止索引越界(虽然 top_n 限制了，但加个保护)
                color_idx = idx % len(current_palette)
                color = current_palette[color_idx]

            ax.bar(years, df_plot[col_name], bottom=bottom, label=label,
                   color=color, width=0.6, edgecolor='white', linewidth=0.05)
            bottom += df_plot[col_name]

        # 2. 装饰子图
        ax.set_title(cfg['title'], loc='left', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xlim(2002.5, 2025.5)
        ax.set_xticks([2003, 2013, 2025])
        ax.tick_params(axis='y', which='both', length=0)
        ax.tick_params(axis='x', length=3, width=0.5)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        # 只在第一列显示 Y 轴标签
        if col == 0:
            ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')
            ax.set_yticklabels(['20', '40', '60', '80'])
            ax.tick_params(axis='y', length=3, width=0.5)
        else:
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        # 3. 子图图例 (每个子图自己带图例，放在底部)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # 分离常规图例和 Others 图例
            main_handles = []
            main_labels = []
            other_handle = None

            for h, l in zip(handles, labels):
                if l == 'Others':
                    other_handle = h
                else:
                    main_handles.append(h)
                    main_labels.append(l)

            # 此时 sorted_handles 先只包含主要污染物
            sorted_handles = list(main_handles)
            sorted_labels = list(main_labels)

            # [核心修改]: 如果存在 Others，根据类别重命名并追加到最后
            if other_handle:
                # 获取当前类别的后缀，例如 "(2)"
                suffix = suffix_map.get(cfg['cat'], '')
                new_label = f"Others{suffix}"

                sorted_handles.append(other_handle)
                sorted_labels.append(new_label)

            # -------------------------------------------------
            # 下面是原本的样式配置代码 (逻辑保持不变，变量名兼容)
            # -------------------------------------------------
            if sorted_handles:
                # 默认字体大小
                leg_fontsize = 10

                # === [场景 A: 图 (b) Single Pollutant] ===
                if 'Single' in cfg['cat']:
                    leg_ncol = 3
                    kwargs = {'loc': 'upper left', 'bbox_to_anchor': (0, -0.08)}

                # === [场景 B: 图 (e, f) 第二行的复杂污染物] ===
                elif 'Four' in cfg['cat']:
                    leg_ncol = 2
                    kwargs = {'loc': 'upper left', 'bbox_to_anchor': (0, -0.08)}
                    leg_fontsize = 10
                elif 'Five' in cfg['cat']:
                    leg_ncol = 1
                    kwargs = {'loc': 'upper left', 'bbox_to_anchor': (0, -0.08)}
                    leg_fontsize = 10

                # === [场景 C: 图 (c, d) 其余第一行的图] ===
                else:
                    leg_ncol = 2
                    kwargs = {'loc': 'upper left', 'bbox_to_anchor': (0, -0.08)}

                # 统一绘制
                ax.legend(sorted_handles, sorted_labels,
                          frameon=False,
                          fontsize=leg_fontsize,
                          handlelength=2,
                          ncol=leg_ncol,
                          columnspacing=1.0,
                          handletextpad=0.4,
                          **kwargs)
# -------------------------------------------------------------------------
# Part 3: 在右下角空白格子 (1, 2) 放置主图例
# -------------------------------------------------------------------------
ax_legend = fig.add_subplot(gs_right[1, 0])
ax_legend.axis('off')  # 隐藏坐标轴
h_main, l_main = ax_main.get_legend_handles_labels()

# 绘制主图例
legend = ax_legend.legend(h_main, l_main,
                          loc='upper left',
                          bbox_to_anchor=(0, 1),
                          title="Pollution Complexity",
                          title_fontsize=14,
                          fontsize=12,
                          frameon=False,
                          labelspacing=1)
legend.get_title().set_fontweight('bold')
plt.savefig("fig4-up-1.png", bbox_inches='tight', dpi=600)
plt.show()