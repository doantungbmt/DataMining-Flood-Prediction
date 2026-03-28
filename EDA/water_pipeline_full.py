"""
Water data analysis pipeline
Steps:
1. Data Cleaning
2. Feature Engineering
3. EDA & Visualization
4. Feature Selection for Flood Prediction

Input:
    /mnt/data/water_data_full_combined.csv

Outputs:
    Multiple CSV files and PNG plots under output_dir

Run:
    python water_pipeline_full.py
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
INPUT_PATH = "data/water_data_full_combined.csv"
OUTPUT_DIR = "data/water_pipeline_outputs"

STATION_COL = "Mã trạm/LakeCode"
STATION_NAME_COL = "Trạm/Hồ"
TIME_RAW_COL = "Thời gian (UTC)"
TIME_COL = "timestamp_utc"
WATER_COL = "Mực nước (m)"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# COMMON HELPERS
# =========================
def save_csv(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path


def save_matrix_csv(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path)
    return path


def save_plot(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================
# STEP 1 - DATA CLEANING
# =========================
def fillna_with_group_median(series: pd.Series) -> pd.Series:
    median_value = series.median(skipna=True)
    if pd.isna(median_value):
        return series
    return series.fillna(median_value)



def interpolate_column_by_station(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    def _interp_one_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(TIME_COL).copy()
        s = g.set_index(TIME_COL)[col]
        s = s.interpolate(method="time", limit_direction="both")
        g[col] = s.to_numpy()
        return g

    return frame.groupby(STATION_COL, group_keys=False).apply(_interp_one_group)



def fill_dynamic_by_station(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    frame = frame.sort_values([STATION_COL, TIME_COL]).copy()
    for col in cols:
        if col not in frame.columns:
            continue
        frame = interpolate_column_by_station(frame, col)
        frame[col] = frame.groupby(STATION_COL)[col].transform(fillna_with_group_median)
        frame[col] = frame[col].fillna(frame[col].median())
    return frame



def fill_static_by_station(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in cols:
        if col not in frame.columns:
            continue
        frame[col] = frame.groupby(STATION_COL)[col].transform(fillna_with_group_median)
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].fillna(frame[col].median())
    return frame



def iqr_outlier_mask_by_station(frame: pd.DataFrame, col: str = WATER_COL) -> pd.Series:
    q1 = frame.groupby(STATION_COL)[col].transform(lambda s: s.quantile(0.25))
    q3 = frame.groupby(STATION_COL)[col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (frame[col] < lower) | (frame[col] > upper)
    return mask.fillna(False)



def step1_data_cleaning(input_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("STEP 1 - DATA CLEANING")

    df_raw = pd.read_csv(input_path, low_memory=False)
    print(f"Raw shape: {df_raw.shape}")

    overview = pd.DataFrame({
        "column": df_raw.columns,
        "dtype": df_raw.dtypes.astype(str).values,
        "null_count": df_raw.isna().sum().values,
        "null_pct": (df_raw.isna().mean() * 100).round(2).values,
        "n_unique": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
    }).sort_values(["null_pct", "null_count"], ascending=False)
    save_csv(overview, "step1_overview_schema_and_missing.csv")

    df = df_raw.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_RAW_COL], errors="coerce", utc=True)

    time_summary = pd.DataFrame({
        "metric": ["time_parse_failures", "min_timestamp_utc", "max_timestamp_utc"],
        "value": [
            int(df[TIME_COL].isna().sum()),
            str(df[TIME_COL].min()),
            str(df[TIME_COL].max()),
        ],
    })
    save_csv(time_summary, "step1_time_summary.csv")

    core_cols = ["type", STATION_COL, STATION_NAME_COL, TIME_RAW_COL, WATER_COL]
    df["missing_core_count"] = df[core_cols].isna().sum(axis=1)
    df["data_quality_flag"] = np.where(df["missing_core_count"] > 0, "missing_core", "ok")
    save_csv(df[["data_quality_flag"]].value_counts().rename("rows").reset_index(), "step1_quality_summary.csv")

    df = df[df["data_quality_flag"] == "ok"].copy()
    lake = df[df["type"] == "Lake"].copy()
    river = df[df["type"] == "River"].copy()

    lake_dynamic = ["Mực nước (m)", "Dung tích (m3)", "Tỷ lệ dung tích (%)", "Q đến (m3/s)", "Q xả (m3/s)"]
    lake_static = ["Dung tích TK (m3)", "Mực nước BT (m)", "Mực nước GC (m)", "x", "y", "province_code", "basin_code"]
    river_dynamic = ["Mực nước (m)", "Chênh lệch cảnh báo (m)"]
    river_static = ["BĐ1 (m)", "BĐ2 (m)", "BĐ3 (m)", "Mực nước lịch sử (m)", "Năm lũ lịch sử", "Cảnh báo value (0-4)", "x", "y"]

    pre_cols = sorted(set(lake_dynamic + lake_static + river_dynamic + river_static))
    pre_nulls = pd.DataFrame({"column": pre_cols})
    pre_nulls["lake_nulls_before"] = pre_nulls["column"].map(lambda c: int(lake[c].isna().sum()) if c in lake.columns else np.nan)
    pre_nulls["river_nulls_before"] = pre_nulls["column"].map(lambda c: int(river[c].isna().sum()) if c in river.columns else np.nan)

    lake["water_outlier_flag"] = iqr_outlier_mask_by_station(lake, WATER_COL)
    river["water_outlier_flag"] = iqr_outlier_mask_by_station(river, WATER_COL)

    lake.loc[lake["water_outlier_flag"], WATER_COL] = np.nan
    river.loc[river["water_outlier_flag"], WATER_COL] = np.nan

    lake = fill_dynamic_by_station(lake, lake_dynamic)
    lake = fill_static_by_station(lake, lake_static)
    river = fill_dynamic_by_station(river, river_dynamic)
    river = fill_static_by_station(river, river_static)

    post_nulls = pd.DataFrame({"column": pre_cols})
    post_nulls["lake_nulls_after"] = post_nulls["column"].map(lambda c: int(lake[c].isna().sum()) if c in lake.columns else np.nan)
    post_nulls["river_nulls_after"] = post_nulls["column"].map(lambda c: int(river[c].isna().sum()) if c in river.columns else np.nan)

    null_compare = pre_nulls.merge(post_nulls, on="column", how="left")
    save_csv(null_compare, "step1_null_comparison_before_after.csv")

    outlier_summary = pd.DataFrame({
        "dataset": ["Lake", "River"],
        "rows": [len(lake), len(river)],
        "water_outliers_flagged": [int(lake["water_outlier_flag"].sum()), int(river["water_outlier_flag"].sum())],
        "outlier_pct": [round(lake["water_outlier_flag"].mean() * 100, 2), round(river["water_outlier_flag"].mean() * 100, 2)],
    })
    save_csv(outlier_summary, "step1_outlier_summary.csv")

    df_step1_clean = pd.concat([lake, river], ignore_index=True)

    final_summary = pd.DataFrame({
        "metric": [
            "raw_rows",
            "rows_after_core_filter",
            "dropped_missing_core",
            "lake_rows_after_cleaning",
            "river_rows_after_cleaning",
            "lake_water_null_after",
            "river_water_null_after",
        ],
        "value": [
            int(df_raw.shape[0]),
            int(df_step1_clean.shape[0]),
            int(df_raw.shape[0] - df_step1_clean.shape[0]),
            int(lake.shape[0]),
            int(river.shape[0]),
            int(lake[WATER_COL].isna().sum()),
            int(river[WATER_COL].isna().sum()),
        ],
    })
    save_csv(final_summary, "step1_final_summary.csv")

    save_csv(df_step1_clean, "water_data_step1_clean.csv")
    save_csv(lake, "water_lake_step1_clean.csv")
    save_csv(river, "water_river_step1_clean.csv")

    print(final_summary.to_string(index=False))
    return df_step1_clean, lake, river


# =========================
# STEP 2 - FEATURE ENGINEERING
# =========================
def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hour"] = frame[TIME_COL].dt.hour
    frame["day"] = frame[TIME_COL].dt.day
    frame["month"] = frame[TIME_COL].dt.month
    frame["dayofweek"] = frame[TIME_COL].dt.dayofweek
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    return frame



def add_exact_hour_lags(frame: pd.DataFrame, lags=(1, 3, 6)) -> pd.DataFrame:
    frame = frame.sort_values([STATION_COL, TIME_COL]).copy()
    base = frame[[STATION_COL, TIME_COL, WATER_COL]].copy()
    for lag in lags:
        lagged = base.copy()
        lagged[TIME_COL] = lagged[TIME_COL] + pd.Timedelta(hours=lag)
        lagged = lagged.rename(columns={WATER_COL: f"water_lag_{lag}h"})
        frame = frame.merge(lagged, on=[STATION_COL, TIME_COL], how="left")
    return frame



def add_rolling_24h(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values([STATION_COL, TIME_COL]).copy()
    parts = []
    for _, g in frame.groupby(STATION_COL, dropna=False):
        g = g.sort_values(TIME_COL).copy()
        s = g.set_index(TIME_COL)[WATER_COL]
        g["water_roll_mean_24h"] = s.rolling("24H", min_periods=1).mean().values
        g["water_roll_std_24h"] = s.rolling("24H", min_periods=2).std().values
        parts.append(g)
    return pd.concat(parts, ignore_index=True)



def infer_typical_interval_hours(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in frame.groupby(STATION_COL):
        g = g.sort_values(TIME_COL)
        diffs = g[TIME_COL].diff().dropna().dt.total_seconds() / 3600
        typical = diffs.mode().iloc[0] if len(diffs) else np.nan
        rows.append({STATION_COL: sid, "typical_interval_hours": typical, "rows": len(g)})
    return pd.DataFrame(rows)



def step2_feature_engineering(df_step1_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("STEP 2 - FEATURE ENGINEERING")

    df = df_step1_clean.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)

    lake = df[df["type"] == "Lake"].copy()
    river = df[df["type"] == "River"].copy()

    candidate_cats = ["Tên tỉnh", "Tên sông", STATION_NAME_COL, "type", STATION_COL]
    available_cats = [c for c in candidate_cats if c in df.columns]

    lake = add_time_features(lake)
    river = add_time_features(river)

    lake = add_exact_hour_lags(lake, lags=(1, 3, 6))
    river = add_exact_hour_lags(river, lags=(1, 3, 6))

    lake = add_rolling_24h(lake)
    river = add_rolling_24h(river)

    lag_coverage = pd.DataFrame({
        "dataset": ["Lake", "River"],
        "rows": [len(lake), len(river)],
        "lag_1h_non_null": [int(lake["water_lag_1h"].notna().sum()), int(river["water_lag_1h"].notna().sum())],
        "lag_3h_non_null": [int(lake["water_lag_3h"].notna().sum()), int(river["water_lag_3h"].notna().sum())],
        "lag_6h_non_null": [int(lake["water_lag_6h"].notna().sum()), int(river["water_lag_6h"].notna().sum())],
    })
    for c in ["lag_1h_non_null", "lag_3h_non_null", "lag_6h_non_null"]:
        lag_coverage[c.replace("_non_null", "_pct")] = (lag_coverage[c] / lag_coverage["rows"] * 100).round(2)
    save_csv(lag_coverage, "step2_lag_coverage.csv")

    rolling_summary = pd.DataFrame({
        "dataset": ["Lake", "River"],
        "roll_mean_non_null": [int(lake["water_roll_mean_24h"].notna().sum()), int(river["water_roll_mean_24h"].notna().sum())],
        "roll_std_non_null": [int(lake["water_roll_std_24h"].notna().sum()), int(river["water_roll_std_24h"].notna().sum())],
        "roll_mean_avg": [round(lake["water_roll_mean_24h"].mean(), 4), round(river["water_roll_mean_24h"].mean(), 4)],
        "roll_std_avg": [round(lake["water_roll_std_24h"].mean(), 4), round(river["water_roll_std_24h"].mean(), 4)],
    })
    save_csv(rolling_summary, "step2_rolling_summary.csv")

    encoding_maps = {}
    combined = pd.concat([lake, river], ignore_index=True)
    for col in available_cats:
        temp = combined[col].astype("string").fillna("__MISSING__")
        cats = pd.Index(sorted(temp.unique()))
        mapping = {cat: idx for idx, cat in enumerate(cats)}
        combined[f"{col}_enc"] = temp.map(mapping).astype(int)
        encoding_maps[col] = pd.DataFrame({col: list(mapping.keys()), f"{col}_enc": list(mapping.values())})
        safe_name = col.replace("/", "_")
        save_csv(encoding_maps[col], f"step2_encoding_map_{safe_name}.csv")

    lake_fe = combined[combined["type"] == "Lake"].copy()
    river_fe = combined[combined["type"] == "River"].copy()

    new_feature_cols = [
        "hour", "day", "month", "dayofweek", "is_weekend",
        "water_lag_1h", "water_lag_3h", "water_lag_6h",
        "water_roll_mean_24h", "water_roll_std_24h",
    ] + [f"{c}_enc" for c in available_cats]

    feature_summary = pd.DataFrame({
        "feature": new_feature_cols,
        "dtype": [str(combined[c].dtype) if c in combined.columns else "N/A" for c in new_feature_cols],
        "null_count": [int(combined[c].isna().sum()) if c in combined.columns else np.nan for c in new_feature_cols],
        "null_pct": [round(combined[c].isna().mean() * 100, 2) if c in combined.columns else np.nan for c in new_feature_cols],
    })
    save_csv(feature_summary, "step2_new_feature_summary.csv")

    lake_interval = infer_typical_interval_hours(lake_fe)
    river_interval = infer_typical_interval_hours(river_fe)
    save_csv(lake_interval, "step2_interval_summary_by_station_lake.csv")
    save_csv(river_interval, "step2_interval_summary_by_station_river.csv")

    interval_summary = pd.DataFrame({
        "dataset": ["Lake", "River"],
        "median_typical_interval_hours": [round(lake_interval["typical_interval_hours"].median(), 2), round(river_interval["typical_interval_hours"].median(), 2)],
        "min_typical_interval_hours": [round(lake_interval["typical_interval_hours"].min(), 2), round(river_interval["typical_interval_hours"].min(), 2)],
        "max_typical_interval_hours": [round(lake_interval["typical_interval_hours"].max(), 2), round(river_interval["typical_interval_hours"].max(), 2)],
    })
    save_csv(interval_summary, "step2_interval_summary.csv")

    save_csv(combined, "water_data_step2_feature_engineered.csv")
    save_csv(lake_fe, "water_lake_step2_feature_engineered.csv")
    save_csv(river_fe, "water_river_step2_feature_engineered.csv")

    print(interval_summary.to_string(index=False))
    return combined, lake_fe, river_fe


# =========================
# STEP 3 - EDA & VISUALIZATION
# =========================
def plot_heatmap(corr_df: pd.DataFrame, title: str, filename: str):
    arr = corr_df.values
    labels = corr_df.columns.tolist()
    plt.figure(figsize=(9, 7))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    save_plot(filename)



def strongest_corr_pairs(corr_df: pd.DataFrame, min_abs_corr: float = 0.5) -> pd.DataFrame:
    rows = []
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if pd.notna(val) and abs(val) >= min_abs_corr:
                rows.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": round(val, 4),
                    "abs_corr": round(abs(val), 4),
                })
    if rows:
        return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    return pd.DataFrame(columns=["feature_1", "feature_2", "corr", "abs_corr"])



def step3_eda_and_visualization(df_step2: pd.DataFrame) -> None:
    print_section("STEP 3 - EDA & VISUALIZATION")

    df = df_step2.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    lake = df[df["type"] == "Lake"].copy()
    river = df[df["type"] == "River"].copy()

    station_variability = (
        df.groupby([STATION_COL, STATION_NAME_COL, "type"])[WATER_COL]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    station_variability["range"] = station_variability["max"] - station_variability["min"]
    station_variability = station_variability.sort_values(["std", "range"], ascending=False)
    top3_stations = station_variability.head(3).copy()
    save_csv(station_variability, "step3_station_variability.csv")
    save_csv(top3_stations, "step3_top3_most_variable_stations.csv")

    for _, row in top3_stations.iterrows():
        sid = row[STATION_COL]
        sname = row[STATION_NAME_COL]
        stype = row["type"]
        g = df[df[STATION_COL] == sid].sort_values(TIME_COL).copy()
        plt.figure(figsize=(12, 5))
        plt.plot(g[TIME_COL], g[WATER_COL])
        plt.title(f"Biến động mực nước theo thời gian - {sname} ({stype})")
        plt.xlabel("Thời gian")
        plt.ylabel("Mực nước (m)")
        plt.xticks(rotation=30)
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in f"{sname}_{stype}")
        save_plot(f"step3_timeseries_{safe_name}.png")

    lake_corr_cols = [c for c in [
        "Mực nước (m)", "Dung tích (m3)", "Tỷ lệ dung tích (%)", "Q đến (m3/s)", "Q xả (m3/s)",
        "Mực nước BT (m)", "Mực nước GC (m)", "water_roll_mean_24h", "water_roll_std_24h"
    ] if c in lake.columns]
    river_corr_cols = [c for c in [
        "Mực nước (m)", "BĐ1 (m)", "BĐ2 (m)", "BĐ3 (m)", "Chênh lệch cảnh báo (m)",
        "Cảnh báo value (0-4)", "water_roll_mean_24h", "water_roll_std_24h"
    ] if c in river.columns]

    lake_corr = lake[lake_corr_cols].corr(numeric_only=True)
    river_corr = river[river_corr_cols].corr(numeric_only=True)

    save_matrix_csv(lake_corr, "step3_lake_corr.csv")
    save_matrix_csv(river_corr, "step3_river_corr.csv")
    plot_heatmap(lake_corr, "Heatmap tương quan - Lake", "step3_heatmap_lake.png")
    plot_heatmap(river_corr, "Heatmap tương quan - River", "step3_heatmap_river.png")

    lake_skew = lake[WATER_COL].dropna().skew()
    river_skew = river[WATER_COL].dropna().skew()
    combined_skew = df[WATER_COL].dropna().skew()
    skew_summary = pd.DataFrame({
        "dataset": ["Lake", "River", "Combined"],
        "rows": [len(lake), len(river), len(df)],
        "water_min": [lake[WATER_COL].min(), river[WATER_COL].min(), df[WATER_COL].min()],
        "water_max": [lake[WATER_COL].max(), river[WATER_COL].max(), df[WATER_COL].max()],
        "water_mean": [lake[WATER_COL].mean(), river[WATER_COL].mean(), df[WATER_COL].mean()],
        "water_std": [lake[WATER_COL].std(), river[WATER_COL].std(), df[WATER_COL].std()],
        "skewness": [lake_skew, river_skew, combined_skew],
    }).round(4)
    save_csv(skew_summary, "step3_skew_summary.csv")

    plt.figure(figsize=(10, 5))
    plt.hist(lake[WATER_COL].dropna(), bins=40)
    plt.title(f"Phân phối mực nước - Lake | skewness = {lake_skew:.3f}")
    plt.xlabel("Mực nước (m)")
    plt.ylabel("Tần suất")
    save_plot("step3_hist_lake.png")

    plt.figure(figsize=(10, 5))
    plt.hist(river[WATER_COL].dropna(), bins=40)
    plt.title(f"Phân phối mực nước - River | skewness = {river_skew:.3f}")
    plt.xlabel("Mực nước (m)")
    plt.ylabel("Tần suất")
    save_plot("step3_hist_river.png")

    lake_strong_corr = strongest_corr_pairs(lake_corr, min_abs_corr=0.5)
    river_strong_corr = strongest_corr_pairs(river_corr, min_abs_corr=0.5)
    save_csv(lake_strong_corr, "step3_lake_strong_correlations.csv")
    save_csv(river_strong_corr, "step3_river_strong_correlations.csv")

    print(top3_stations[[STATION_COL, STATION_NAME_COL, "type", "std", "range"]].to_string(index=False))


# =========================
# STEP 4 - FEATURE SELECTION
# =========================
def target_correlation_table(frame: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        s = frame[[col, target_col]].copy().dropna()
        if len(s) == 0 or s[col].nunique(dropna=True) <= 1:
            corr = np.nan
        else:
            corr = s[col].corr(s[target_col])
        rows.append({
            "feature": col,
            "corr_with_target": corr,
            "abs_corr_with_target": abs(corr) if pd.notna(corr) else np.nan,
            "non_null_rows": len(s),
        })
    return pd.DataFrame(rows).sort_values("abs_corr_with_target", ascending=False)



def run_rf_feature_importance(frame: pd.DataFrame, target_col: str, feature_cols: list[str], random_state: int = 42):
    X = frame[feature_cols].copy()
    y = frame[target_col].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
    )
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics_df = pd.DataFrame({
        "metric": ["roc_auc", "average_precision", "positive_rate_test"],
        "value": [auc, ap, y_test.mean()],
    })
    return importance_df, metrics_df



def step4_feature_selection(df_step2: pd.DataFrame) -> None:
    print_section("STEP 4 - FEATURE SELECTION FOR FLOOD PREDICTION")

    df = df_step2.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    lake = df[df["type"] == "Lake"].copy()
    river = df[df["type"] == "River"].copy()

    target_availability = pd.DataFrame({
        "dataset": ["Lake", "River"],
        "rows": [len(lake), len(river)],
        "BĐ3_non_null": [
            int(lake["BĐ3 (m)"].notna().sum()) if "BĐ3 (m)" in lake.columns else 0,
            int(river["BĐ3 (m)"].notna().sum()) if "BĐ3 (m)" in river.columns else 0,
        ],
        "BĐ3_non_null_pct": [
            round(lake["BĐ3 (m)"].notna().mean() * 100, 2) if "BĐ3 (m)" in lake.columns else 0.0,
            round(river["BĐ3 (m)"].notna().mean() * 100, 2) if "BĐ3 (m)" in river.columns else 0.0,
        ],
    })
    save_csv(target_availability, "step4_target_availability.csv")

    river["is_flood"] = (river[WATER_COL] > river["BĐ3 (m)"]).astype(int)
    lake["is_high_water_lake"] = (lake[WATER_COL] > lake["Mực nước BT (m)"]).astype(int)

    target_summary = pd.DataFrame({
        "dataset": ["River", "Lake"],
        "target_name": ["is_flood", "is_high_water_lake"],
        "positive_rows": [int(river["is_flood"].sum()), int(lake["is_high_water_lake"].sum())],
        "total_rows": [len(river), len(lake)],
        "positive_rate_pct": [round(river["is_flood"].mean() * 100, 2), round(lake["is_high_water_lake"].mean() * 100, 2)],
    })
    save_csv(target_summary, "step4_target_summary.csv")

    lake_candidate_features = [
        "hour", "day", "month", "dayofweek", "is_weekend",
        "water_lag_1h", "water_lag_3h", "water_lag_6h",
        "water_roll_mean_24h", "water_roll_std_24h",
        "Dung tích (m3)", "Tỷ lệ dung tích (%)", "Q đến (m3/s)", "Q xả (m3/s)",
        "Mực nước GC (m)", "x", "y", "province_code", "basin_code",
        "Tên tỉnh_enc", "Trạm/Hồ_enc", "Mã trạm/LakeCode_enc",
    ]
    lake_candidate_features = [c for c in lake_candidate_features if c in lake.columns]

    river_candidate_features = [
        "hour", "day", "month", "dayofweek", "is_weekend",
        "water_lag_1h", "water_lag_3h", "water_lag_6h",
        "water_roll_mean_24h", "water_roll_std_24h",
        "Mực nước lịch sử (m)", "Năm lũ lịch sử", "Cảnh báo value (0-4)",
        "x", "y", "Tên tỉnh_enc", "Trạm/Hồ_enc", "Mã trạm/LakeCode_enc",
    ]
    river_candidate_features = [c for c in river_candidate_features if c in river.columns]

    feature_inventory = pd.DataFrame({
        "dataset": ["Lake", "River"],
        "feature_count": [len(lake_candidate_features), len(river_candidate_features)],
        "features": [", ".join(lake_candidate_features), ", ".join(river_candidate_features)],
    })
    save_csv(feature_inventory, "step4_feature_inventory.csv")

    lake_corr_target = target_correlation_table(lake, "is_high_water_lake", lake_candidate_features)
    river_corr_target = target_correlation_table(river, "is_flood", river_candidate_features)
    save_csv(lake_corr_target, "step4_lake_corr_with_target.csv")
    save_csv(river_corr_target, "step4_river_corr_with_target.csv")

    lake_importance, lake_metrics = run_rf_feature_importance(lake, "is_high_water_lake", lake_candidate_features)
    river_importance, river_metrics = run_rf_feature_importance(river, "is_flood", river_candidate_features)
    save_csv(lake_importance, "step4_lake_rf_importance.csv")
    save_csv(river_importance, "step4_river_rf_importance.csv")
    save_csv(lake_metrics, "step4_lake_rf_metrics.csv")
    save_csv(river_metrics, "step4_river_rf_metrics.csv")

    top_n = 12
    lake_top = lake_importance.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(lake_top["feature"], lake_top["importance"])
    plt.title("Top feature importance - Lake (proxy target)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("step4_lake_feature_importance.png")

    river_top = river_importance.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(river_top["feature"], river_top["importance"])
    plt.title("Top feature importance - River (is_flood)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot("step4_river_feature_importance.png")

    lake_recommended_core = [
        "Tỷ lệ dung tích (%)",
        "Dung tích (m3)",
        "water_roll_mean_24h",
        "month",
        "Mực nước GC (m)",
        "Q đến (m3/s)",
        "Q xả (m3/s)",
        "x",
        "y",
        "basin_code",
        "province_code",
    ]
    lake_recommended_optional = [
        "dayofweek",
        "day",
        "water_roll_std_24h",
        "Tên tỉnh_enc",
        "Trạm/Hồ_enc",
    ]
    lake_not_recommended_now = [
        "water_lag_1h",
        "water_lag_3h",
        "water_lag_6h",
        "hour",
        "is_weekend",
        "Mã trạm/LakeCode_enc",
    ]

    recommendation_table = pd.DataFrame({
        "group": (
            ["core"] * len(lake_recommended_core) +
            ["optional"] * len(lake_recommended_optional) +
            ["not_recommended_now"] * len(lake_not_recommended_now)
        ),
        "feature": lake_recommended_core + lake_recommended_optional + lake_not_recommended_now,
        "reason": (
            [
                "Phản ánh trạng thái đầy hồ, importance cao",
                "Biến thủy văn cốt lõi của hồ",
                "Bắt xu hướng ngắn hạn của hồ",
                "Nắm yếu tố mùa vụ",
                "Ngưỡng vận hành có ý nghĩa vật lý",
                "Phản ánh lưu lượng đến hồ",
                "Phản ánh vận hành xả hồ",
                "Thông tin vị trí địa lý",
                "Thông tin vị trí địa lý",
                "Đặc trưng lưu vực",
                "Đặc trưng hành chính/địa phương",
            ] +
            [
                "Có thể giữ để bổ sung chu kỳ thời gian",
                "Có thể hữu ích theo mùa/ngày",
                "Đo độ dao động gần đây",
                "Mã hóa vùng địa lý",
                "Hữu ích nếu mô hình theo từng hồ",
            ] +
            [
                "Hồ chủ yếu ghi nhận theo ngày nên lag 1h hầu như rỗng",
                "Hồ chủ yếu ghi nhận theo ngày nên lag 3h hầu như rỗng",
                "Hồ chủ yếu ghi nhận theo ngày nên lag 6h rất ít dữ liệu",
                "Giá trị thấp vì hồ đo theo ngày, giờ ít ý nghĩa",
                "Tín hiệu yếu trong dữ liệu hiện tại",
                "Dễ overfit theo mã hồ nếu mục tiêu là tổng quát hóa",
            ]
        ),
    })
    save_csv(recommendation_table, "step4_lake_feature_recommendation.csv")

    lake_final_cols = [
        TIME_COL, STATION_NAME_COL, "Tên tỉnh", "type", "is_high_water_lake"
    ] + lake_recommended_core + lake_recommended_optional
    lake_final_cols = [c for c in lake_final_cols if c in lake.columns or c == "is_high_water_lake"]
    lake_final_dataset = lake[lake_final_cols].copy()
    save_csv(lake_final_dataset, "water_lake_step4_feature_selected.csv")

    final_conclusion = pd.DataFrame({
        "item": [
            "original_target_for_river",
            "proxy_target_for_lake",
            "river_positive_rate_pct",
            "lake_proxy_positive_rate_pct",
            "lake_most_important_feature_1",
            "lake_most_important_feature_2",
            "lake_most_important_feature_3",
            "main_lake_modeling_note",
        ],
        "value": [
            "is_flood = 1 if water > BĐ3",
            "is_high_water_lake = 1 if water > Mực nước BT",
            round(river["is_flood"].mean() * 100, 2),
            round(lake["is_high_water_lake"].mean() * 100, 2),
            lake_importance.iloc[0]["feature"],
            lake_importance.iloc[1]["feature"],
            lake_importance.iloc[2]["feature"],
            "Do hồ chủ yếu ghi nhận 24h, nên ưu tiên rolling + storage + inflow/outflow + vị trí; không ưu tiên lag 1h/3h/6h",
        ],
    })
    save_csv(final_conclusion, "step4_final_conclusion.csv")

    print(target_summary.to_string(index=False))
    print("\nTop 5 Lake features:")
    print(lake_importance.head(5).to_string(index=False))


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT_PATH}")

    df_step1_clean, lake_step1, river_step1 = step1_data_cleaning(INPUT_PATH)
    df_step2, lake_step2, river_step2 = step2_feature_engineering(df_step1_clean)
    step3_eda_and_visualization(df_step2)
    step4_feature_selection(df_step2)

    print_section("PIPELINE COMPLETED")
    print(f"Tất cả file kết quả đã lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
