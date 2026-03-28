# CHAPTER 1: INTRODUCTION TO THE PROBLEM
## 1.1. Background and research problem
In the context of increasingly complex climate change, flooding is becoming one of the natural disasters causing severe damage in Vietnam, especially in the Central region. Narrow, steep terrain and a system of short rivers cause floodwaters to concentrate quickly, increasing the risk of flash floods and causing difficulties for forecasting.
Traditional forecasting methods still have many limitations such as long processing times, large resource requirements, and difficulty integrating multi-source data. In addition, actual data is often missing and noisy, affecting the accuracy of the model.
Therefore, the project aims to build an intelligent flood forecasting system capable of processing multi-source data, forecasting accurately, and supporting real-time alerts.

## 1.2. Proposed solution and contributions
The project builds a complete pipeline from data to application, including gathering, processing, and standardizing data from multiple different sources.
The XGBoost model is used to forecast water levels thanks to its ability to process non-linear data efficiently. The model is optimized and evaluated using indicators such as RMSE and MAE to ensure accuracy.
Besides, a visual web system is developed to display data and provide real-time alerts via a backend API, ensuring performance and scalability.

The project's contributions:
- **Scientific**: Combining data processing and machine learning in flood forecasting.
- **Technical**: Building a real-time forecasting system.
- **Practical**: Supporting early warning and minimizing damages.

# CHAPTER 2: OVERVIEW OF RELATED STUDIES
## 2.1. Hydrological and hydraulic models
Physics-based models play an important role in flood forecasting due to their ability to directly simulate natural processes and provide high interpretability. However, these models often require massive computational resources and heavily depend on input data.
Some typical models include:
- HEC-RAS: 1D/2D hydrodynamic simulation, used to build highly accurate inundation maps.
- MIKE 11: Combined with numerical weather prediction (NWP) to extend early warning time.
- SVSMR: Semi-distributed model, well adapted to mountainous basins.
- SFM: Rainfall-runoff model, used in early warning systems.

Overall, these models have high accuracy but are limited in processing speed and scalability.

## 2.2. Machine learning and deep learning methods
Machine learning and deep learning methods are increasingly widely applied due to their ability to process non-linear relationships without detailed physical process modeling.
Some typical models:
- LSTM, BiLSTM: Suitable for time-series data, capable of remembering long-term dependencies.
- ConvLSTM: Combining spatio-temporal features, effective in urban flood simulation.
- KANs: Improving non-linear modeling capabilities compared to traditional networks.
- KELM: Optimized using evolutionary algorithms to enhance forecasting performance.

These methods have high accuracy but depend on the quality and completeness of the data.

## 2.3. Hybrid models and research trends
The current trend is combining physical models and machine learning to leverage the advantages of both.
Some typical approaches:
- Combining physical models and AI: Calibrating physical model errors to improve accuracy.
- VMD-BiLSTM: Reducing data noise prior to forecasting.
- TimeGAN: Generating synthetic data to supplement areas lacking data.
- SSPP: Probabilistic forecasting and optimizing flood peaks.
- P2M: Rapidly building inundation maps with high performance.

Hybrid models help balance accuracy, speed, and real-world applicability.

## 2.4. Remarks and research gaps
From previous studies, we can draw the following:
- Physical models have good interpretability but are slow and difficult to scale.
- Machine learning models perform highly but are data-dependent.
- Hybrid models are the main development trend.

However, gaps still exist:
- Lack of a fully integrated system from data to application.
- Difficulty processing missing and noisy data in reality.
- Limitations in deploying real-time forecasting systems.

Therefore, this project proposes building a flood forecasting system based on XGBoost, combined with a data processing pipeline and a visual web system to enhance practical applicability.

# CHAPTER 3: DATA SOURCES AND PROCESSING WORKFLOW
## 3.1. Data sources
In this study, data was collected from multiple sources to fully reflect the characteristics of the hydrological system for the flood forecasting problem. The main data sources include:
- Hydrometeorological observation data, providing information on rainfall, water levels, and environmental factors over time.
- Reservoir operation data, including capacity, inflow, and outflow.
- Water level data at river and lake stations.

The dataset, after aggregation, is stored in tabular form with a large number of records over a time series, establishing a foundation for the analysis process and creating the forecasting model.

## 3.2. Data preprocessing workflow
Actual data often has issues such as missing data, noise, and outliers. Therefore, the preprocessing step is performed to ensure the quality of input data for the model.
- **Handling missing data**: Missing values are handled depending on the data type, where categorical variables are filled with the most common value (mode), while continuous variables are replaced with the median to limit the impact of outliers.
- **Handling outliers**: Abnormal values are processed using the percentile capping method, helping remove noise while retaining the majority of the data distribution.
- **Data standardization**: Applying Min-Max Scaling to bring continuous variables to the same [0,1] range, improving the model's learning efficiency.
- **Time synchronization**: Data from multiple sources are aligned along the time axis to ensure the continuity of the data series, suitable for forecasting problems.

## 3.3. Feature engineering
To enhance forecasting capability, new features are engineered from the original data:
- Time variable (Month) to reflect seasonality.
- Rolling Mean (7 days) helps smooth data and reduce short-term noise.
- Delta_1d represents the upward or downward trend of water levels over time.
- The target variable is defined by shifting the water level value to the next time step (t+1), serving the forecasting problem.

## 3.4. Data splitting
Data is split chronologically to ensure the reality of the problem:
- 80% of the initial data is used for training.
- 20% of the latter data is used for testing.

This split helps the model learn from past data and forecast the future, accurately reflecting real-world scenarios.

# CHAPTER 4: DATA ANALYSIS AND RESULTS
## 4.1. Data scale and geographical distribution
*Figure 4.1: Data scale and geographical distribution*

The dataset consists of 57,474 records with 27 features, divided into river data (14,024) and lake data (43,450). This large scale ensures representativeness of hydrological characteristics.
Data distribution is uneven across regions, with Gia Lai accounting for the largest share (>25,000 records), followed by Da Nang and Quang Ngai (~10,000 each).
Certain lakes, like Dinh Binh, have more complete data, indicating differences between stations and affecting model training efficiency.

## 4.2. Missing data analysis
### 4.2.1. Reservoir data
Analyzing reservoir data reveals a significant missing rate in some important variables (Figure 4.2). Specifically:
- The outflow variable is missing about 40%; this parameter relates directly to water discharge operations and often encounters logging issues.
- The inflow variable is missing about 21%, affecting the ability to model the water balance.
- Variables related to location and current states, such as coordinates, station codes, and current water levels, are complete.

Generally, reservoir data is relatively high quality for core variables, but operational variables still require careful handling.

### 4.2.2. River data
Unlike lake data, river data has many entirely missing variables (100%), including capacity, inflow, and outflow, due to river stations lacking reservoir structures (Figure 4.3).
Additionally, historical water level variables are missing by about 45%, posing difficulties for long-term trend analysis. However, warning variables like Alarm 1, 2, 3 (BD1, BD2, BD3) have very low missing rates (<10%), ensuring reliability in assigning alarm status labels.

## 4.3. Time series analysis
### 4.3.1. Data series length
Analyzing the length of the data series reveals clear differences among lakes. Some lakes, like Dinh Binh, have data dating back to 2013, providing a long and stable series for model training (Figure 4.4).
Conversely, some lakes only have recent data (e.g., from 2024), leading to short datasets and limitations in applying deep learning models.
A cohort of lakes with consistent data during the 2015-2026 period (such as SeSan4, Ialy, Kanak) is considered suitable for building multi-station forecasting models.

### 4.3.2. Temporal distribution
Analyzing the interval between records demonstrates that data is collected over fixed cycles, primarily daily. This ensures stability and continuity of the time series, fitting for forecasting models (Figure 4.5).

## 4.4. Data distribution analysis
### 4.4.1. Water level distribution
The water level distribution is right-skewed, where the majority of values sit at a low level (normal state), while high values (major floods) appear infrequently.
Furthermore, there are some extremely large values (e.g., >20,000m) identified as data errors. These values must be removed during preprocessing to prevent model distortion.

### 4.4.2. Outlier analysis
Outlier analysis shows some clusters of unusually high values (~1150), particularly at stations like SeSan. These are not entirely errors but reflect differences in terrain characteristics and scales across regions (Figure 4.7).
Thus, it's necessary to standardize or group data by region to maintain model consistency.

## 4.5. Trend and seasonality analysis
### 4.5.1. Monthly variations
Analyzing by month reveals clear seasonality (Figure 4.8):
- Water levels peak in February (~292m)
- They drop to their lowest in July-September (~273m)

This reflects characteristic wet-dry cycles and forms a critical basis for engineering the "Month" feature in the model.

### 4.5.2. Annual variations
Average water levels show a slight upward trend over time, while fluctuations exist due to seasonal cycles and climate change impacts (Figure 4.9).

## 4.6. Typical time series analysis
*Figure 4.10: SeSan4 station time series*

The time series at SeSan4 station shows (Figure 4.10):
- High data continuity.
- Strong fluctuations over time.
- A 7-day Rolling Mean helps smooth the data and highlights trends.

These characteristics indicate the data is apt for short-term and medium-term forecasting models.

## 4.7. Correlation analysis between features
### 4.7.1. Reservoir data
The correlation matrix shows that water level (water_level) strongly correlates with variables like normal_level and dead_level, reflecting multicollinearity among features describing physical traits (Figure 4.11).
Also, inflow and outflow show a distinct positive correlation, aligning with reservoir operations. The delta_level_1d variable also has a positive correlation with water level, proving to be an essential feature for the model to learn fluctuation trends.

### 4.7.2. River data
For river data, water level correlates very strongly with warning levels BD1, BD2, BD3, illustrating that alert thresholds depend directly on water levels (Figure 4.12).
The warning_gap variable shows a clear negative correlation, reflecting an inverse relationship: the higher the water level, the less the safety margin.

## 4.8. Analysis by station and data status
Results show data heavily concentrates in the (very_high, very_high) state between inflow and water level, reflecting operational scenarios during flood seasons (Figure 4.13).
From this analysis, a few processing directions are proposed:
- Remove absurd outliers.
- Separate river and lake data.
- Add features like Rolling Mean, Delta, and Month.

## 4.9. Reservoir clustering
Clustering results split data into two main groups (Figure 4.14):
- Stable group: Lakes with similar traits and low volatility.
- Highly volatile group: Lakes heavily impacted by inflow/outflow.

A few lakes like Binh Dien sit apart, demonstrating distinct characteristics, potentially due to missing data or interpolation methods.

# CHAPTER 5: FORECASTING MODEL AND RESULT EVALUATION
## 5.1. Overview of related studies and achieved results
In the field of flood forecasting, many studies have proposed deep learning, hybrid, and physical models to boost forecasting accuracy.
The TW-TimeGAN-BiLSTM model by Chen et al. (2025) achieved RMSE = 7.67, MAE = 3.88, and NSE = 0.903, but performance dropped with longer forecast horizons. Lee et al. (2025)'s hybrid method combining SFM and AI achieved R² = 0.92 and notably improved peak flood errors.
According to Aghenda et al. (2025), deep learning models like LSTM, CNN, and ANN generally outperform traditional methods. Meanwhile, physical models like MIKE11 and HEC-RAS gain high accuracy but are relatively difficult to deploy in real time.
Additionally, modern models like CNN-LSTM-KANs, VMD-BiLSTM, and MESAO-KELM consistently show good performance, especially when combining multiple data processing techniques.

## 5.2. Proposed forecasting model (XGBoost)
In this study, the XGBoost machine learning model was selected for water level prediction due to its capability to efficiently handle non-linear data and suitability for tabular data.

### 5.2.1. Feature engineering
Input features are engineered from time-series data, including:
- Time variable (Month) to reflect seasonality.
- Rolling Mean (7 days) designed to smooth the data.
- Delta_1d revealing the trend of water level fluctuations.

The target variable is determined by shifting the water level one time step forward (t+1).

### 5.2.2. Model training and evaluation
Data is chronologically split:
- Train: 80% (46,844 samples)
- Test: 20% (11,712 samples)

The model was assessed via RMSE and MAE, achieving:
- RMSE = 0.0190
- MAE = 0.0017

Results indicate highly accurate predictions on the normalized data.

## 5.3. Comparison with previous studies
The XGBoost model attains lower errors compared to several prior studies. However, it must be noted that the data was normalized and the task was short-term forecasting, precluding direct comparisons with models utilizing physical units or long-term predictions.
Overall, XGBoost exhibits high efficiency, is easy to deploy, and proves suitable for real-world data, whereas deep learning and hybrid models are typically more complex and demand larger resources.
