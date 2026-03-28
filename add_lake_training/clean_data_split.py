import os
import pandas as pd
import numpy as np

def main():
    input_file = '../water_data_full_combined.csv'
    output_dir = '.'
    
    print("Loading original dataset...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print("Cleaning 'type' column...")
    if 'type' in df.columns:
        df['type_lower'] = df['type'].astype(str).str.strip().str.lower()
        
        # Map back to standardized 'Lake' and 'River'
        df.loc[df['type_lower'].str.contains('lake', na=False), 'type_clean'] = 'Lake'
        df.loc[df['type_lower'].str.contains('river', na=False), 'type_clean'] = 'River'
        
        # Drop rows where type is neither Lake nor River
        df = df.dropna(subset=['type_clean'])
        
        # Split datasets
        lake_df = df[df['type_clean'] == 'Lake'].copy()
        river_df = df[df['type_clean'] == 'River'].copy()
        
        lake_df['type'] = 'Lake'
        river_df['type'] = 'River'
        
        lake_df = lake_df.drop(columns=['type_clean', 'type_lower'])
        river_df = river_df.drop(columns=['type_clean', 'type_lower'])
    else:
        print("Error: 'type' column not found!")
        return

    print(f"Split complete: {len(lake_df)} Lake records, {len(river_df)} River records.")
    
    # ---------------------------------------------
    # 1. CLEANING LAKE DATA
    # ---------------------------------------------
    print("Cleaning Lake dataset (Focusing on 'Tỷ lệ dung tích (%)')...")
    
    # Drop columns that are completely empty for Lake
    lake_df = lake_df.dropna(axis=1, how='all')
    
    if 'Tỷ lệ dung tích (%)' in lake_df.columns:
        # Fill missing with median
        median_tl = lake_df['Tỷ lệ dung tích (%)'].median()
        lake_df['Tỷ lệ dung tích (%)'] = lake_df['Tỷ lệ dung tích (%)'].fillna(median_tl)
        
        # Capping 1st and 99th percentile for safety
        p01 = lake_df['Tỷ lệ dung tích (%)'].quantile(0.01)
        p99 = lake_df['Tỷ lệ dung tích (%)'].quantile(0.99)
        lake_df.loc[lake_df['Tỷ lệ dung tích (%)'] < p01, 'Tỷ lệ dung tích (%)'] = p01
        lake_df.loc[lake_df['Tỷ lệ dung tích (%)'] > p99, 'Tỷ lệ dung tích (%)'] = p99

    # Basic cleaning for other columns in lake_df
    for col in lake_df.select_dtypes(include=[np.number]).columns:
        if col != 'Tỷ lệ dung tích (%)':
            lake_df[col] = lake_df[col].fillna(lake_df[col].median())
    for col in lake_df.select_dtypes(exclude=[np.number]).columns:
        lake_df[col] = lake_df[col].fillna("Unknown")

    # ---------------------------------------------
    # 2. CLEANING RIVER DATA
    # ---------------------------------------------
    print("Cleaning River dataset (Comprehensive cleaning)...")
    
    # Drop columns that are completely empty for River
    river_df = river_df.dropna(axis=1, how='all')
    
    num_cols = river_df.select_dtypes(include=[np.number]).columns
    cat_cols = river_df.select_dtypes(exclude=[np.number]).columns

    # Fill numerical with median per station if possible, else global median
    for col in num_cols:
        global_median = river_df[col].median()
        if 'Trạm/Hồ' in river_df.columns:
            # fill with groupby transform
            river_df[col] = river_df.groupby('Trạm/Hồ')[col].transform(lambda x: x.fillna(x.median()))
        
        # Fill remaining NaNs with global median
        river_df[col] = river_df[col].fillna(global_median)
        
        # Capping outliers (1st to 99th percentile)
        p01 = river_df[col].quantile(0.01)
        p99 = river_df[col].quantile(0.99)
        river_df.loc[river_df[col] < p01, col] = p01
        river_df.loc[river_df[col] > p99, col] = p99

    for col in cat_cols:
        river_df[col] = river_df[col].fillna("Unknown")

    # Save outputs
    print(f"Saving to {output_dir}...")
    if 'Thời gian (UTC)' in lake_df.columns:
        lake_df = lake_df.sort_values(by=['Trạm/Hồ', 'Thời gian (UTC)'])
    if 'Thời gian (UTC)' in river_df.columns:
        river_df = river_df.sort_values(by=['Trạm/Hồ', 'Thời gian (UTC)'])

    lake_df.to_csv(os.path.join(output_dir, 'lake_cleaned.csv'), index=False)
    river_df.to_csv(os.path.join(output_dir, 'river_cleaned.csv'), index=False)
    
    print("\n=== Combining Datasets ===")
    combined_df = pd.concat([lake_df, river_df], axis=0, ignore_index=True)
    combined_file = os.path.join(output_dir, 'water_data_cleaned_combined.csv')
    combined_df.to_csv(combined_file, index=False)
    print(f"Saved combined data to {combined_file}")
    
    print("\n=== Verification ===")
    print(f"Lake Total Missing Values: {lake_df.isnull().sum().sum()}")
    print(f"River Total Missing Values: {river_df.isnull().sum().sum()}")
    print(f"Combined Data Shape: {combined_df.shape}")
    print("Data cleaning successfully completed.")

if __name__ == '__main__':
    main()
