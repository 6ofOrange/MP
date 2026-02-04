import geopandas as gpd
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
df = pd.read_csv("./data/compound_1to6_county2024.csv")
file = "./data/china_metrics_2003.gpkg"
gdf = gpd.read_file(file, layer="pw_metrics") \
    .to_crs("EPSG:4326")
columns_to_select = [
    'XIANMIAN_',
    'PYNAME',
    'geometry'
]
df_index = gdf[columns_to_select]
merged_gdf = gpd.GeoDataFrame(pd.merge(df_index, df, on=['XIANMIAN_'], how='inner'))
pic_gdf = merged_gdf[['XIANMIAN_', 'PYNAME', 'geometry', 'k1_2013', 'k2_2013', 'k3_2013', 'k4_2013', 'k5_2013', 'k6_2013', 'k1_2018', 'k2_2018', 'k3_2018', 'k4_2018', 'k5_2018', 'k6_2018', 'k1_2025', 'k2_2025', 'k3_2025', 'k4_2025', 'k5_2025', 'k6_2025']]
pic_gdf2 = pic_gdf.copy()
pic_gdf2['k0_2013'] = (
    365
    - pic_gdf2['k1_2013']
    - pic_gdf2['k2_2013']
    - pic_gdf2['k3_2013']
    - pic_gdf2['k4_2013']
    - pic_gdf2['k5_2013']
    - pic_gdf2['k6_2013']
)
pic_gdf2['k0_2018'] = (
    365
    - pic_gdf2['k1_2018']
    - pic_gdf2['k2_2018']
    - pic_gdf2['k3_2018']
    - pic_gdf2['k4_2018']
    - pic_gdf2['k5_2018']
    - pic_gdf2['k6_2018']
)
pic_gdf2['k0_2025'] = (
    365
    - pic_gdf2['k1_2025']
    - pic_gdf2['k2_2025']
    - pic_gdf2['k3_2025']
    - pic_gdf2['k4_2025']
    - pic_gdf2['k5_2025']
    - pic_gdf2['k6_2025']
)

col_to_cate = {
    'k1_2013': 1,
    'k2_2013': 2,
    'k3_2013': 3,
    'k4_2013': 4,
    'k5_2013': 5,
    'k6_2013': 6,
    'k0_2013': 0
}
# 提取要比较的列
target_cols = list(col_to_cate.keys())
# 找到每行中最大值所在的列名
max_col = pic_gdf2[target_cols].idxmax(axis=1)
# 映射为类别编号
pic_gdf2['cate'] = max_col.map(col_to_cate)

# 图4下
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib as mpl
from matplotlib.colors import ListedColormap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
nine_dash_shp_path = "./data/shpfiles/china_nine_dotted_line.shp"
nine = gpd.read_file(nine_dash_shp_path, encoding='utf-8')

meige_1 = pic_gdf2.copy()
years = [2013, 2018, 2025]
sub_labels = ['(g)', '(h)', '(i)']
mapping = {
    0: "Clean",
    1: "Single Pollutant",
    2: "Two Pollutants",
    3: "Three Pollutants",
    4: "Four Pollutants",
    5: "Five Pollutants"

}

meige_1["category"] = meige_1["cate"].map(mapping)

categories = [
    "Clean",
    "Single Pollutant",
    "Two Pollutants",
    "Three Pollutants",
    "Four Pollutants",
    "Five Pollutants"
]

colors = {
    "Clean": "#F7F7F7",  # 极淡灰白 (几乎透明背景)
    "Single Pollutant": "#FFFFB2",  # 淡黄 (开始有迹象)
    "Two Pollutants": "#FED976",  # 暖黄
    "Three Pollutants": "#FEB24C",  # 橙黄
    "Four Pollutants": "#FD8D3C",  # 深橙
    "Five Pollutants": "#F03B20",  # 鲜红
    # "Six Pollutants": "#BD0026",   # 深褐红 (极端)
}
meige_1["category"] = pd.Categorical(
    meige_1["category"],
    categories=categories,
    ordered=True
)

cmap = ListedColormap([colors[c] for c in categories])


def add_south_china_sea_inset_dynamic(main_ax, china_shp, nine_dash_shp):
    ax_ins = main_ax.inset_axes([0.90, -0.07, 0.18, 0.36],
                                projection=ccrs.LambertConformal(central_latitude=90, central_longitude=115))

    # 设置子图范围
    ax_ins.set_extent([105, 125, 2, 23], crs=ccrs.PlateCarree())

    # 绘制中国边界
    china_shp.plot(ax=ax_ins, facecolor='none', edgecolor='black', linewidth=0.3, transform=ccrs.PlateCarree(),
                   zorder=1)

    # 绘制九段线
    nine_dash_shp.plot(ax=ax_ins, facecolor='none', edgecolor='gray', linewidth=0.6, linestyle='--',
                       transform=ccrs.PlateCarree(), zorder=2)

    # 关闭多余元素
    ax_ins.spines['geo'].set_linewidth(0.5)  # 边框变细

    return ax_ins


china_shp_path = "./data/shpfiles/china_country.shp"
nine_dash_shp_path = "./data/shpfiles/china_nine_dotted_line.shp"
china_boundary = gpd.read_file(china_shp_path, encoding='utf-8')
fig = plt.figure(figsize=(18, 8))
proj = ccrs.LambertConformal(central_latitude=90, central_longitude=105)

for i, year in enumerate(years):
    # --- A. 数据处理 ---
    # 动态构建该年份的列名映射
    col_to_cate = {f'k{k}_{year}': k for k in range(1, 6)}
    col_to_cate[f'k0_{year}'] = 0  # 加上Clean列

    target_cols = list(col_to_cate.keys())

    # 复制数据并计算最大值
    current_gdf = pic_gdf2.copy()
    # 找出每行最大值对应的列名
    max_col = current_gdf[target_cols].idxmax(axis=1)
    # 映射为 0-6 的数字
    current_gdf['cate'] = max_col.map(col_to_cate)
    # 映射为 文本类别
    current_gdf["category"] = current_gdf["cate"].map(mapping)
    # 转为有序分类变量
    current_gdf["category"] = pd.Categorical(
        current_gdf["category"], categories=categories, ordered=True
    )
    # --- B. 创建子图 ---
    ax = fig.add_subplot(1, 3, i + 1, projection=proj)
    # ax.set_frame_on(False)
    # 设置范围
    ax.set_extent([80, 127, 17, 54], crs=ccrs.PlateCarree())

    # 添加标号 (g) 2013 ...
    label_text = f"{sub_labels[i]} {year}"
    ax.text(0.03, 0.97, label_text, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_frame_on(False)
    # --- 绘制地图 ---
    # 1. 主数据
    current_gdf.plot(
        column="category",
        categorical=True,
        cmap=cmap,
        linewidth=0.1,
        edgecolor="gray",
        legend=False,
        ax=ax,
        transform=ccrs.PlateCarree(),
        zorder=1
    )
    # 2. 国界
    china_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6, transform=ccrs.PlateCarree(),
                        zorder=4)
    # 3. 九段线
    nine.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6, linestyle='--', transform=ccrs.PlateCarree(),
              zorder=3)

    # --- 添加图例 (只在第一个图添加) ---

    if i == 0:
        handles = [mpatches.Patch(color=colors[c], label=c) for c in categories]
        legend = ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.01, -0.05),
            fontsize=10,
            title_fontsize=11,
            frameon=False
        )

    # --- F. 添加南海子图 ---
    add_south_china_sea_inset_dynamic(ax, china_boundary, nine)

plt.tight_layout()
plt.savefig("fig4_bottom.png", dpi=600, bbox_inches='tight')
plt.show()