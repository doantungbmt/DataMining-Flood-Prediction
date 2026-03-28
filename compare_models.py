import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Loading preprocessed data for testing...")
df = pd.read_csv("water_data_preprocessed.csv")
lake_df = df[df['type'].str.contains('Lake', case=False, na=False)].copy()

if 'Thời gian (UTC)' in lake_df.columns:
    lake_df['Thời gian (UTC)'] = pd.to_datetime(lake_df['Thời gian (UTC)'])
    lake_df = lake_df.sort_values(by=['Trạm/Hồ', 'Thời gian (UTC)']).reset_index(drop=True)

    # Feature Engineering exactly as in training
    lake_df['Month'] = lake_df['Thời gian (UTC)'].dt.month
    lake_df['Rolling_Mean_7d'] = lake_df.groupby('Trạm/Hồ')['Mực nước (m)'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    lake_df['Delta_1d'] = lake_df.groupby('Trạm/Hồ')['Mực nước (m)'].diff(1).fillna(0)
    lake_df['Target_Muc_Nuoc_t_plus_1'] = lake_df.groupby('Trạm/Hồ')['Mực nước (m)'].shift(-1)
    
    lake_df = lake_df.dropna(subset=['Target_Muc_Nuoc_t_plus_1'])

    features = ['Mực nước (m)', 'Month', 'Rolling_Mean_7d', 'Delta_1d', 
                'Dung tích (m3)', 'Q đến (m3/s)', 'Q xả (m3/s)']
    target = 'Target_Muc_Nuoc_t_plus_1'

    lake_df = lake_df.dropna(subset=features)
    lake_df = lake_df.sort_values(by='Thời gian (UTC)')
    
    # 80-20 Train-Test split
    split_idx = int(len(lake_df) * 0.8)
    test_df = lake_df.iloc[split_idx:]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Testing on {len(X_test)} samples.")
    
    print("Loading XGBoost Model...")
    xgb_model = joblib.load("xgboost_flood_model.pkl")
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    
    print("Loading LightGBM Model...")
    lgb_model = joblib.load("lightgbm_flood_model.pkl")
    lgb_pred = lgb_model.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    
    print("\n" + "="*40)
    print("MODEL COMPARISON (XGBoost vs LightGBM)")
    print("="*40)
    print(f"| Metric | XGBoost   | LightGBM  |")
    print(f"|--------|-----------|-----------|")
    print(f"| RMSE   | {xgb_rmse:9.6f} | {lgb_rmse:9.6f} |")
    print(f"| MAE    | {xgb_mae:9.6f} | {lgb_mae:9.6f} |")
    print("="*40 + "\n")
    
else:
    print("Column 'Thời gian (UTC)' not found.")
