import os
import glob
import numpy as np
import pandas as pd

POL_LIST = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
UPPER_LIMITS = {
    "PM2.5": 2000,
    "PM10": 2500,
    "CO": 90,
    "NO2": 900,
    "SO2": 1500,
    "O3": 1200,
}

def cal_row_col(lon0, lat0, res, lon, lat):
    col = np.around((lon - lon0) / res).astype(int)
    row = np.around((lat - lat0) / (-res)).astype(int)
    return col, row

def collect_npy_by_year(x_root, nan_txt_root):
    files = []
    for year_dir in sorted(os.listdir(x_root)):
        if not year_dir.isdigit():
            continue
        year = int(year_dir)
        nan_file = os.path.join(nan_txt_root, f"{year}_y_nan.txt")
        bad = set()
        if os.path.isfile(nan_file):
            with open(nan_file) as f:
                bad = {os.path.splitext(l.strip())[0] for l in f}
        for f in glob.glob(os.path.join(x_root, year_dir, "*.npy")):
            if os.path.splitext(os.path.basename(f))[0] not in bad:
                files.append((year, f))
    return files

def read_data(npy_file, year, y_root):
    data = np.load(npy_file)

    Tc = data[:, 1] - 273.15
    Tdc = data[:, 0] - 273.15
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    RH = 100 * es(Tdc) / es(Tc)
    data = np.hstack((data, RH[:, None]))

    X = np.lib.stride_tricks.sliding_window_view(data, 48, axis=0)
    X = np.moveaxis(X, -1, 1)

    lon = float(os.path.basename(npy_file).split("_")[0])
    lat = float(os.path.basename(npy_file).split("_")[1].split("n")[0][:-1])
    col, row = cal_row_col(70, 60, 0.1, lon, lat)
    key = f"{col}_{row}"

    station_dir = os.path.join(y_root, str(year))
    csv = glob.glob(os.path.join(station_dir, f"*{key}*.csv"))[0]

    df = pd.read_csv(csv).drop(columns=["Unnamed: 0"])
    for p in POL_LIST:
        df[p] = df[p].clip(upper=UPPER_LIMITS[p])

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").resample("h").interpolate().ffill().bfill()

    y = np.column_stack([df[p].values[47:] for p in POL_LIST]).astype(np.float32)
    return X.astype(np.float32), y
