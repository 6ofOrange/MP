#图5数据预处理
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
df = pd.read_csv("./data/compound_1to6_county2024.csv")
# 定义时间段
periods = {
    '2003_2012': (2003, 2012),
    '2013_2017': (2013, 2017),
    '2018_2025': (2018, 2025)
}
# 获取所有数据列（排除 XIANMIAN_）
data_cols = [col for col in df.columns if col != 'XIANMIAN_']
# 构建一个映射：category -> list of (year, column_name)
category_year_cols = defaultdict(list)
for col in data_cols:
    parts = col.split('_')
    if len(parts) == 2 and parts[0].startswith('k') and parts[0][1:].isdigit():
        category = parts[0]
        year = int(parts[1])
        category_year_cols[category].append((year, col))
# 对每个类别和每个时间段计算平均值，并添加新列
for category, year_col_list in category_year_cols.items():
    # 转为字典便于查找，但其实不需要；我们直接按年份筛选
    year_to_col = {year: col for year, col in year_col_list}
    for period_name, (start, end) in periods.items():
        cols_in_period = [year_to_col[y] for y in range(start, end + 1) if y in year_to_col]
        if cols_in_period:  # 确保该时间段有列存在
            new_col_name = f"{category}_{period_name}"
            df[new_col_name] = df[cols_in_period].mean(axis=1)
        else:
            # 可选：如果时间段无数据，可以赋 NaN 或跳过
            df[f"{category}_{period_name}"] = pd.NA

# 最终 df 包含原始列 + 18 个新列（6 类别 × 3 时段）
categories = [f'k{i}' for i in range(1, 7)]  # ['k1', 'k2', ..., 'k6']
period_names = ['2003_2012', '2013_2017', '2018_2025']
new_cols = [f"{cat}_{period}" for cat in categories for period in period_names]
# 确保这些列确实存在于 df 中（可选，用于容错）
existing_new_cols = [col for col in new_cols if col in df.columns]
# 创建新 DataFrame：包含 'XIANMIAN_' 和新增的18列
df_1 = df[['XIANMIAN_'] + existing_new_cols].copy()

file = "./data/china_metrics_2003.gpkg"
gdf = gpd.read_file(file, layer="pw_metrics") \
    .to_crs("EPSG:4326")
columns_to_select = [
    'XIANMIAN_',
    'NAME',
    'PYNAME',
    'geometry'
]
df_index = gdf[columns_to_select]
merged_gdf = gpd.GeoDataFrame(pd.merge(df_index, df_1, on='XIANMIAN_', how='inner'))

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

china_shp_path = "./data/shpfiles/china_country.shp"
nine_dash_shp_path = "./data/shpfiles/china_nine_dotted_line.shp"
meige_1 = merged_gdf.copy()
nine = gpd.read_file(nine_dash_shp_path, encoding='utf-8')

# ================= 颜色与样式设置 =================
abs_vmin = 0
abs_vmax = 200
abs_ticks = [0, 50, 100, 150, 200]
diff_configs = [
    # k1 (Single):
    {"ticks": [-20, 0, 40, 80, 120]},
    # k2
    {"ticks": [-100, -50, 0, 50, 100]},
    # k3
    {"ticks": [-100, -50, 0, 50, 100]},
    # k4
    {"ticks": [-150, -100, -50, 0, 20]},
    # k5:
    {"ticks": [-120, -80, -40, 0, 20]}
]
norm_abs = colors.Normalize(vmin=abs_vmin, vmax=abs_vmax)
cmap_abs = "GnBu"
colors_pos = ['white', '#EF767B']  # 正值：白→红
colors_neg = ['white', '#43A3EF']  # 负值：白→蓝紫
colors2 = colors_neg[::-1] + colors_pos[1:]
cmap_2 = LinearSegmentedColormap.from_list('diverging_cmap', colors2, N=256)


# ================= 定义南海子图函数 (使用 inset_axes) =================
def add_south_china_sea_inset_relative(parent_ax, china_shp, nine_dash_gdf):
    """
    在给定的 parent_ax 中，使用相对坐标插入南海子图
    """
    # 相对坐标位置 [x, y, width, height]，相对于 parent_ax 的大小
    # 0.82, 0.02 表示放在右下角
    ax_ins = parent_ax.inset_axes([0.90, -0.07, 0.18, 0.36],
                                  projection=ccrs.LambertConformal(central_latitude=90, central_longitude=115))

    # 设置子图范围
    ax_ins.set_extent([105, 125, 2, 23], crs=ccrs.PlateCarree())

    # 绘制中国边界
    china_shp.plot(ax=ax_ins, facecolor='none', edgecolor='black', linewidth=0.3, transform=ccrs.PlateCarree(),
                   zorder=1)

    # 绘制九段线
    nine_dash_gdf.plot(ax=ax_ins, facecolor='none', edgecolor='gray', linewidth=0.6, linestyle='--',
                       transform=ccrs.PlateCarree(), zorder=2)

    # 关闭多余元素
    ax_ins.spines['geo'].set_linewidth(0.5)  # 边框变细

    return ax_ins


# 预先读取中国轮廓，避免在循环中重复读取以提高速度
china_outline = gpd.read_file(china_shp_path)

# ================= 主绘图循环 =================

# 定义行列结构
rows = 5  # k1 到 k5
cols = 4  # 三个时间段
k_labels = ["Single Pollutant", "Two Pollutants", "Three Pollutants", "Four Pollutants", "Five Pollutants"]
time_periods = ["2003_2012", "2013_2017", "2018_2025"]  # 对应列名后缀
time_titles = ["2003-2012", "2013-2017", "2018-2025", "Unclean Days"]  # 显示在图顶部的标题
# k_labels = ["Single Pollutant"]
# time_periods = ["2003_2012","2013_2017", "2018_2023"] # 对应列名后缀
# time_titles = ["2003-2012", "2013-2017", "2018-2023"]  # 显示在图顶部的标题
row_letters = ['a', 'b', 'c', 'd', 'e']
# 创建画布
# fig, axes = plt.subplots(rows, cols, figsize=(20,24),
#                         subplot_kw={'projection': ccrs.LambertConformal(central_latitude=90, central_longitude=105)})
fig = plt.figure(figsize=(20, 24))
gs_outer = fig.add_gridspec(1, 2, width_ratios=[3.05, 1], wspace=0.22,
                            left=0.05, right=0.95, top=0.95, bottom=0.05)

# 3. 定义内层网格
# 左边块：5行3列，wspace=0.02 (非常紧密，符合你的要求)
gs_left = gs_outer[0].subgridspec(rows, 3, wspace=0.02, hspace=0.03)

# 右边块：5行1列
gs_right = gs_outer[1].subgridspec(rows, 1, hspace=0.03)

# 4. 初始化 axes 数组 (保持和你原有代码兼容的 5x4 结构)
axes = np.empty((rows, 4), dtype=object)
# 调整子图间距
# plt.subplots_adjust(wspace=0.1, hspace=0.05, left=0.05, right=0.95)
for r in range(rows):
    # --- 生成前3列 (位于 gs_left) ---
    for c in range(3):
        axes[r, c] = fig.add_subplot(
            gs_left[r, c],
            projection=ccrs.LambertConformal(central_latitude=90, central_longitude=105)
        )

    # --- 生成第4列 (位于 gs_right) ---
    # 注意这里用 gs_right[r, 0] 因为右边块只有1列
    axes[r, 3] = fig.add_subplot(
        gs_right[r, 0],
        projection=ccrs.LambertConformal(central_latitude=90, central_longitude=105)
    )
for i in range(rows):
    k_num = i + 1  # k1, k2...
    d_cfg = diff_configs[i]
    for j in range(cols):
        ax = axes[i, j]
        # 1. 设定地图显示范围
        ax.set_extent([80, 127, 17, 54], crs=ccrs.PlateCarree())
        if j < 3:
            period = time_periods[j]
            col_name = f"k{k_num}_{period}"  # 构造列名，如 k1_2003_2012
            plot_data = meige_1[col_name]
            current_cmap = cmap_abs
            current_norm = norm_abs
            is_diff = False
        else:
            col_late = f"k{k_num}_2018_2025"
            col_early = f"k{k_num}_2013_2017"
            plot_data = meige_1[col_late] - meige_1[col_early]
            vmin = plot_data.min()
            if i == 4:
                vmax = 30
            else:
                vmax = plot_data.max()
            divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            current_cmap = cmap_2
            current_norm = divnorm
            is_diff = True
        meige_1.plot(
            column=plot_data,
            cmap=current_cmap,
            norm=current_norm,
            linewidth=0.1,  # 设置为0去除县界描边，使画面更干净
            edgecolor="gray",  # 如果一定要描边，建议设为0.1或更细
            legend=False,
            ax=ax,
            transform=ccrs.PlateCarree(),
            missing_kwds={"color": "lightgray", "label": "Missing"}
        )
        china_outline.plot(
            ax=ax,
            facecolor='none',  # 透明填充
            edgecolor='black',  # 黑色边框
            linewidth=0.4,  # 【关键】国界线加粗
            transform=ccrs.PlateCarree(),
            zorder=4  # 放在最顶层，盖住区县线
        )
        # 3. 叠加九段线和必要的省界（如果 meige_1 是县级数据，建议这里另外叠加一个只有省界的图层）
        nine.plot(ax=ax,
                  facecolor='none',
                  edgecolor='black',
                  linewidth=0.5,
                  linestyle='--',
                  transform=ccrs.PlateCarree(),
                  zorder=3)
        label_text = f"({row_letters[i]}{j + 1})"
        ax.text(0.04, 0.96, label_text,
                transform=ax.transAxes,  # 使用相对坐标 (0-1)
                ha='left', va='top',  # 左上对齐
                fontsize=14,  # 字号稍大一点以示区分
                fontweight='bold',
                fontname='Arial',
                zorder=10)
        # 4. 添加南海小图
        add_south_china_sea_inset_relative(ax, china_outline, nine)

        # 5. 装饰：去除经纬度刻度（保持整洁）
        ax.axis('off')  # 或者使用 ax.set_xticks([]) 等

        # 6. 添加列标题 (仅第一行)
        if i == 0:
            ax.set_title(time_titles[j], fontsize=16, pad=14, fontname='Arial', fontweight='bold')

    # ================= 每行添加一个图例 (Top Tier Style) =================
    # 添加图例
    # --- 图例 1:  (放在第3列右侧) ---
    cax = axes[i, 2].inset_axes([1.17, 0.1, 0.05, 0.9])  # [x, y, width, height] 相对坐标
    sm = mpl.cm.ScalarMappable(cmap=cmap_abs, norm=norm_abs)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", ticks=abs_ticks)
    cbar.ax.tick_params(labelsize=12)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Arial')
        l.set_fontname('Arial')

    # --- 图例 2: 差值 (放在第4列右侧) ---
    cax2 = axes[i, 3].inset_axes([1.17, 0.1, 0.05, 0.9])
    sm2 = mpl.cm.ScalarMappable(cmap=cmap_2, norm=divnorm)
    sm2._A = []
    cbar2 = fig.colorbar(sm2, cax=cax2, orientation="vertical", ticks=d_cfg["ticks"])
    cbar2.ax.tick_params(labelsize=12)
    for l in cbar2.ax.yaxis.get_ticklabels():
        l.set_family('Arial')
        l.set_fontname('Arial')

    # 仅在中间行的图例上写 Unit，或者每一行都写
    if i == 2:
        cbar.set_label("Unclean Days", rotation=270, labelpad=15, fontsize=14, fontname='Arial')
        cbar2.set_label("Unclean Days", rotation=270, labelpad=15, fontsize=14, fontname='Arial')

    # ================= 添加行标题 (左侧) =================
    # 利用每行第一个子图的坐标系添加文本
    axes[i, 0].text(-0.1, 0.5, k_labels[i],
                    transform=axes[i, 0].transAxes,
                    rotation=90, va='center', ha='right',
                    fontsize=16, fontweight='bold', fontname='Arial')
# 保存图片
plt.savefig('fig5.png', dpi=300, bbox_inches='tight')
plt.show()