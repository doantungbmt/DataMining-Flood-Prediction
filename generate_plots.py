import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # Safe fallback

# 1. Load Data
print("Loading data...")
df = pd.read_csv("water_data_full_combined.csv", low_memory=False)

# 2. Phân bổ Dữ liệu (Data Distribution)
print("Plotting Data Distribution...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
plt.title("Ty le du lieu Song vs Ho")
plt.ylabel("")

plt.subplot(1, 2, 2)
top_provinces = df['Tên tỉnh'].value_counts().head(10)
sns.barplot(x=top_provinces.values, y=top_provinces.index, palette="viridis")
plt.title("Top 10 Tinh/Thanh pho co nhieu du lieu nhat")
plt.xlabel("So luong ban ghi")
plt.tight_layout()
plt.savefig("1_data_distribution.png", dpi=300)
plt.close()

# 3. Tỷ lệ Thiếu Dữ liệu (Missing Values)
print("Plotting Missing Values...")
lake_df = df[df['type'].str.contains('Lake', case=False, na=False)]
river_df = df[df['type'].str.contains('River', case=False, na=False)]

missing_lake = lake_df.isnull().mean() * 100
missing_river = river_df.isnull().mean() * 100

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
if not missing_lake[missing_lake > 0].empty:
    missing_lake[missing_lake > 0].sort_values().plot.barh(color='lightcoral')
plt.title("Ty le thieu du lieu - Ho (%)")
plt.xlabel("% Thieu")

plt.subplot(1, 2, 2)
if not missing_river[missing_river > 0].empty:
    missing_river[missing_river > 0].sort_values().plot.barh(color='lightskyblue')
plt.title("Ty le thieu du lieu - Song (%)")
plt.xlabel("% Thieu")
plt.tight_layout()
plt.savefig("2_missing_data_rates.png", dpi=300)
plt.close()

# 4. Biểu đồ Tần suất Mực nước
print("Plotting Water Level Histogram...")
plt.figure(figsize=(10, 6))
# Filter out extreme outliers for better visualization of the distribution
valid_water_level = df[(df['Mực nước (m)'] > 0) & (df['Mực nước (m)'] < 1000)]['Mực nước (m)'].dropna()
sns.histplot(valid_water_level, bins=50, kde=True, color='teal')
plt.title("Phan phoi Tan suat Muc nuoc (m)")
plt.xlabel("Muc nuoc (m)")
plt.ylabel("Tan suat")
plt.savefig("3_water_level_histogram.png", dpi=300)
plt.close()

# 5. Biến động Mực nước Trung bình theo Tháng
print("Plotting Monthly Avg Water Level...")
if 'Thời gian (UTC)' in df.columns:
    df['Thời gian (UTC)'] = pd.to_datetime(df['Thời gian (UTC)'], errors='coerce')
    df['Tháng'] = df['Thời gian (UTC)'].dt.month
    monthly_avg = df.groupby('Tháng')['Mực nước (m)'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=monthly_avg, x='Tháng', y='Mực nước (m)', palette="Blues_d")
    plt.plot(monthly_avg['Tháng'] - 1, monthly_avg['Mực nước (m)'], color='red', marker='o', linewidth=2)
    plt.title("Bien dong Muc nuoc trung binh theo thang")
    plt.xlabel("Thang")
    plt.ylabel("Muc nuoc trung binh (m)")
    plt.savefig("4_monthly_avg_water_level.png", dpi=300)
    plt.close()

# 6. Biến động Chuỗi thời gian: Trạm SeSan4
print("Plotting SeSan4 Time Series...")
df['temp_name'] = df['Trạm/Hồ'].astype(str).str.replace(' ', '').str.lower()
sesan4 = df[df['temp_name'].str.contains('sesan4', na=False)].copy()

if not sesan4.empty and 'Thời gian (UTC)' in sesan4.columns:
    sesan4 = sesan4.sort_values(by='Thời gian (UTC)')
    sesan4 = sesan4.set_index('Thời gian (UTC)')
    sesan4['Mực nước (m)'] = pd.to_numeric(sesan4['Mực nước (m)'], errors='coerce')
    sns_plot_data = sesan4['Mực nước (m)'].dropna()
    
    # Let's resample daily mean if data is too dense
    sns_plot_data = sns_plot_data.resample('D').mean().dropna()
    rolling_mean = sns_plot_data.rolling(window=7, min_periods=1).mean()
    
    plt.figure(figsize=(14, 6))
    plt.plot(sns_plot_data.index, sns_plot_data.values, label='Muc nuoc thuc te', alpha=0.5, color='royalblue')
    plt.plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean 7 Days', color='darkorange', linewidth=2)
    plt.title("Bien dong Muc nuoc tram SeSan4")
    plt.xlabel("Thoi gian")
    plt.ylabel("Muc nuoc (m)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("5_sesan4_time_series.png", dpi=300)
    plt.close()
else:
    print("Warning: SeSan4 data not found or no time column.")

# 7. Ma trận Tương quan (Correlation Matrix)
print("Plotting Correlation Matrices...")
# Lake
numeric_lake = lake_df.select_dtypes(include=[np.number])
if not numeric_lake.empty:
    corr_lake = numeric_lake.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_lake, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Ma Tran Tuong Quan - Du Lieu Ho")
    plt.tight_layout()
    plt.savefig("6_correlation_matrix_lake.png", dpi=300)
    plt.close()

# River
numeric_river = river_df.select_dtypes(include=[np.number])
if not numeric_river.empty:
    corr_river = numeric_river.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_river, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Ma Tran Tuong Quan - Du Lieu Song")
    plt.tight_layout()
    plt.savefig("7_correlation_matrix_river.png", dpi=300)
    plt.close()

print("All plots generated successfully!")
