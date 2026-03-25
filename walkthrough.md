# Data Pre-Processing Walkthrough

We have successfully pre-processed the [water_data_full_combined.csv](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/water_data_full_combined.csv) dataset according to the guidelines provided in the course PDFs (`Course Overview and Introduction to Data Mining.pdf` & `Data Pre-Processing.pdf`).

## 1. Handling Missing Values
Several columns had a high percentage of missing values.
- **Categorical & Discrete columns**: `Mã Cảnh báo`, `province_code`, `basin_code`, `Cảnh báo value (0-4)` were filled with their respective **mode** (most frequent value).
- **Text columns**: `Cảnh báo/Xu thế` was filled with a global constant `"Unknown"`.
- **Continuous numerical columns**: e.g., `Mực nước`, `Dung tích`, `Q đến`, `Q xả` were filled using the **median** value of each column (to avoid distortion from extreme outliers).

*Verification*: The total number of missing values was reduced from >100,000 to **0**.

## 2. Handling Outliers (Smoothing Noise)
Exploratory Data Analysis revealed severe outliers (e.g., `Mực nước` reaching 20,280 m).
- **Z-score style Percentile Capping**: We handled outliers by capping the continuous variables at the **1st and 99th percentiles**. This removes the extreme noise while retaining 98% of valid statistical distributions, providing a stable foundation for predictive models.

## 3. Data Normalization
- **Min-Max Scaling**: Applied Min-Max Normalization to continuous numerical features (such as `Mực nước`, `Dung tích`, etc.) to scale them into the `[0, 1]` range. 
*Verification*: Descriptive statistics confirm that `min = 0.0` and `max = 1.0` for all target columns.

## Deliverables
- **Preprocessed Script**: [preprocess_data.py](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/preprocess_data.py)
- **Cleaned Dataset**: `water_data_preprocessed.csv`

The data is now clean, free of missing values, and scaled properly for downstream mining tasks like clustering, classification, or regression.

## 4. Data Visualizations (EDA)
We generated 7 statistical charts to reproduce the visualizations in `Data mining.pdf` using the raw dataset.

````carousel
![Phân bổ Dữ liệu Sông/Hồ và Tỉnh/Thành phố](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/1_data_distribution.png)
<!-- slide -->
![Tỷ lệ thiếu dữ liệu của trạm Hồ và Sông](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/2_missing_data_rates.png)
<!-- slide -->
![Biểu đồ tần suất phân phối Mực nước](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/3_water_level_histogram.png)
<!-- slide -->
![Biến động mức nước trung bình theo từng tháng](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/4_monthly_avg_water_level.png)
<!-- slide -->
![Sự biến động chuỗi thời gian của Mực nước tại trạm SeSan4](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/5_sesan4_time_series.png)
<!-- slide -->
![Heatmap Tương quan dữ liệu Hồ](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/6_correlation_matrix_lake.png)
<!-- slide -->
![Heatmap Tương quan dữ liệu Sông](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/7_correlation_matrix_river.png)
````

## 5. Flood Prediction Modeling (XGBoost)
We successfully trained an `XGBRegressor` model to predict the next time-step normalized `Mực nước (m)` for lake stations based on chronological features.

### Feature Engineering
- **Time feature**: Created `Month` column to capture seasonal trends.
- **Statistical features**: Added an automated 7-day `Rolling Mean` to smooth short-term noise and a `Delta_1d` (first derivative) to capture the trend of rising or falling water levels, per station.
- **Target**: Shifted `Mực nước (m)` by 1 time step to create the target variable.

### Training & Evaluation
- Set an explicit chronological `Train/Test` split: 80% (older data) for training (`46,844` samples), 20% (newer data) for testing (`11,712` samples).
- **Performance**:
  - `RMSE`: **0.0190**
  - `MAE`: **0.0017**
  - The ultra-low errors indicate that the model performs exceedingly well on predicting the normalized water level. We visualize this excellent fit via a comparison of actual vs. predicted values.

### Results Visualizations
````carousel
![XGBoost Feature Importance](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/xgb_feature_importance.png)
<!-- slide -->
![XGBoost Actual vs Predicted](/home/tungnd/.gemini/antigravity/brain/a38a0b65-9a06-4d5e-b622-1c09559404cb/xgb_actual_vs_predicted.png)
````

## 6. Backend API Service (FastAPI)
To facilitate easy deployment for your web application, we engineered a fast, reliable Backend API using **FastAPI** to serve the trained XGBoost model.

### API Capabilities
- **Endpoint**: `POST /predict`
- **Payload Schema**: Accepts a clean JSON with all 7 necessary normalized features (e.g. `muc_nuoc`, `month`, `rolling_mean_7d`, `delta_1d`, etc.) validated using Pydantic.
- **Output**: Returns the exact predicted water level (`predicted_muc_nuoc_t_plus_1`).

### Deliverables
All service files are contained within the `service_predict/` directory:
- [main.py](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/service_predict/main.py): The core FastAPI application logic.
- [schemas.py](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/service_predict/schemas.py): Input/Output API type checking and validation.
- [requirements.txt](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/service_predict/requirements.txt): Python dependencies explicitly versioned.
- [Dockerfile](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/service_predict/Dockerfile): Containerization script allowing immediate Docker deployment (`docker build -t flood_api .`).
- [test_api.py](file:///home/tungnd/SDH/CS2207.CH203_Data-mining/project_final/service_predict/test_api.py): A convenience script to test the `/predict` endpoint live.

*Verification*: Local HTTP ping tests confirmed the server loads the `.pkl` model efficiently and accurately returns valid prediction JSONs under ~50ms responses.
