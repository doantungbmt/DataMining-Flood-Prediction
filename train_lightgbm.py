import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

print("Loading preprocessed data...")
df = pd.read_csv("water_data_preprocessed.csv")

# Only keep Lake data and specific columns to avoid memory/data leakage issues
lake_df = df[df['type'].str.contains('Lake', case=False, na=False)].copy()

if 'Thời gian (UTC)' in lake_df.columns:
    lake_df['Thời gian (UTC)'] = pd.to_datetime(lake_df['Thời gian (UTC)'])
    # Sort chronologically
    lake_df = lake_df.sort_values(by=['Trạm/Hồ', 'Thời gian (UTC)']).reset_index(drop=True)

    print("Feature Engineering...")
    # 1. Month Encoding
    lake_df['Month'] = lake_df['Thời gian (UTC)'].dt.month
    
    # 2. Calculate rolling mean and delta per station
    lake_df['Rolling_Mean_7d'] = lake_df.groupby('Trạm/Hồ')['Mực nước (m)'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    lake_df['Delta_1d'] = lake_df.groupby('Trạm/Hồ')['Mực nước (m)'].diff(1).fillna(0)
    
    # Target: Predict next step's Water Level -> shift(-1)
    lake_df['Target_Muc_Nuoc_t_plus_1'] = lake_df.groupby('Trạm/Hồ')['Mực nước (m)'].shift(-1)
    
    # Drop rows where target is NaN (the last timestamp per station)
    lake_df = lake_df.dropna(subset=['Target_Muc_Nuoc_t_plus_1'])

    # Features to train on
    features = ['Mực nước (m)', 'Month', 'Rolling_Mean_7d', 'Delta_1d', 
                'Dung tích (m3)', 'Q đến (m3/s)', 'Q xả (m3/s)']
    target = 'Target_Muc_Nuoc_t_plus_1'

    # Drop nan in features
    lake_df = lake_df.dropna(subset=features)
    
    # Chronological sort for splitting
    lake_df = lake_df.sort_values(by='Thời gian (UTC)')
    
    # 80-20 Train-Test split
    split_idx = int(len(lake_df) * 0.8)
    
    train_df = lake_df.iloc[:split_idx]
    test_df = lake_df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:\n- RMSE: {rmse:.4f}\n- MAE: {mae:.4f}")
    
    # Visualizations
    print("Plotting Feature Importance...")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=len(features), importance_type='split', ax=plt.gca(), color='teal')
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig("lgb_feature_importance.png", dpi=300)
    plt.close('all')
    
    print("Plotting Actual vs Predicted...")
    plt.figure(figsize=(14, 6))
    # Take a specific station for actual vs predicted to see the trend clearly over time
    example_station = test_df['Trạm/Hồ'].value_counts().idxmax()
    subset_test = test_df[test_df['Trạm/Hồ'] == example_station].sort_values(by='Thời gian (UTC)')
    subset_pred = model.predict(subset_test[features])
    
    plt.plot(subset_test['Thời gian (UTC)'], subset_test[target].values, label='Actual Muc Nuoc', alpha=0.7, color='royalblue')
    plt.plot(subset_test['Thời gian (UTC)'], subset_pred, label='Predicted Muc Nuoc', alpha=0.7, color='darkorange', linewidth=2)
    plt.title(f"Actual vs. Predicted Water Level (Test Subset - {example_station})")
    plt.xlabel("Thoi gian (UTC)")
    plt.ylabel("Normalized Water Level")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lgb_actual_vs_predicted.png", dpi=300)
    plt.close()
    
    print("Saving model for deployment...")
    joblib.dump(model, "lightgbm_flood_model.pkl")
    print("Model saved to lightgbm_flood_model.pkl")
    
    print("Modeling completed successfully!")
else:
    print("Column 'Thời gian (UTC)' not found. Cannot proceed with temporal operations.")
