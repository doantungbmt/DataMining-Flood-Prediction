import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def preprocess_data(input_file, output_file):
    print("Loading data...")
    # Read the dataset with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(input_file, low_memory=False)
    
    # 1. Handling Missing Values
    print("Handling Missing Values...")
    
    # Text columns
    text_cols = ['Cảnh báo/Xu thế']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            
    # Categorical/Discrete numerical columns
    cat_num_cols = ['Mã Cảnh báo', 'province_code', 'basin_code', 'Cảnh báo value (0-4)']
    for col in cat_num_cols:
        if col in df.columns:
            # fill with mode, if mode exists
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna(0)

    # Continuous numerical columns
    # Identifying continuous columns by excluding text, categorical, and non-numeric columns
    exclude_cols = ['type', 'Mã trạm/LakeCode', 'Trạm/Hồ', 'Tên sông/Lưu vực', 'Tên tỉnh', 'Thời gian (UTC)'] + text_cols + cat_num_cols
    cont_num_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    for col in cont_num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    print(f"Missing values handled. Remaining total missing values: {df.isnull().sum().sum()}")
    
    # 2. Handling Outliers
    print("Handling Outliers by Capping at 1st and 99th percentiles...")
    for col in cont_num_cols:
        # Calculate 1st and 99th percentiles
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        # Cap values
        df.loc[df[col] < p01, col] = p01
        df.loc[df[col] > p99, col] = p99

    # 3. Data Transformation & Normalization
    print("Normalizing data using Min-Max Scaler...")
    scaler = MinMaxScaler()
    df[cont_num_cols] = scaler.fit_transform(df[cont_num_cols])
    
    # Verify outputs
    print("\n=== Pre-processed Data Info ===")
    df.info()
    
    print("\n=== Normalized Columns Describe ===")
    print(df[cont_num_cols].describe())
    
    print(f"Saving pre-processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    input_csv = "water_data_full_combined.csv"
    output_csv = "water_data_preprocessed.csv"
    preprocess_data(input_csv, output_csv)
