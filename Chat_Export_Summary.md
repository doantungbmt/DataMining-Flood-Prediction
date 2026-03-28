# Tổng hợp Quá trình Dự án Data Mining: Dự báo Lũ lụt

Dưới đây là tài liệu tổng hợp và ghi chép lại toàn bộ quá trình làm việc giữa Người dùng và Trợ lý ảo (AI Agent) trong việc xây dựng hệ thống mô hình dự báo lũ lụt bằng Machine Learning.

---

## 1. Phân tích Yêu cầu & Khám phá Dữ liệu (EDA)
- **Nguồn dữ liệu:** `water_data_full_combined.csv` (dữ liệu thủy văn của các trạm Sông và Hồ thủy điện).
- **Yêu cầu giáo trình:** Dựa trên các file `.pdf` bài giảng môn Data Mining (CS2207.CH203), áp dụng các phương pháp tiền xử lý chuẩn mực.
- **Vấn đề phát hiện:** Dữ liệu thô chứa nhiều Missing values (đặc biệt ở các cột cảnh báo, mức nước lịch sử) và Outliers cực đoan (Mực nước có lúc lên tới >20,000m).

## 2. Tiền xử lý dữ liệu đợt đầu (Pre-processing)
- Lấp missing values: Sử dụng **Median** cho dữ liệu số và **Mode** cho dữ liệu phân loại.
- Loại bỏ nhiễu: Áp dụng thuật toán giới hạn (Percentile Capping ở mốc 1% và 99%) để cắt các đỉnh nhiễu vô lý.
- Chuẩn hóa dữ liệu: Scale toàn bộ về dải `[0, 1]` bằng thuật toán `MinMaxScaler`.

## 3. Trực quan hóa Dữ liệu (Data Visualization)
- Vẽ 7 biểu đồ thống kê cơ bản giống hệt trong slide bài giảng môn học bằng thư viện `matplotlib` và `seaborn`:
  1. Phân bổ dữ liệu Sông/Hồ và top tỉnh thành.
  2. Tỷ lệ thiếu dữ liệu của 2 loại trạm.
  3. Histogram tần suất mực nước.
  4. Mực nước trung bình thay đổi theo tháng.
  5. Biến động chuỗi thời gian của Trạm SeSan4 (với Rolling Mean 7 ngày).
  6. Ma trận tương quan (Heatmap) Hồ.
  7. Ma trận tương quan (Heatmap) Sông.

## 4. Huấn luyện Mô hình XGBoost (XGBRegressor)
- Xây dựng file `train_xgboost.py` để tạo mô hình Máy học (XGBoost) dự báo mực nước tương lai `(t+1)`.
- **Feature Engineering**: Tạo thêm cột `Month` (bắt mùa vụ), `Rolling Mean 7 Days` (làm mịn tín hiệu) và `Delta 1 Day` (đo gia tốc nước).
- **Phân tách Dữ liệu (Split)**: Chia tập train/test theo đúng *dòng thời gian* (80% quá khứ để Train, 20% tương lai để Test). Tránh Data Leakage.
- **Đánh giá**:
  - `RMSE`: 0.0190
  - `MAE`: 0.0017
- Lưu trữ mô hình thành file `.pkl` (đóng gói bằng `joblib`) sẵn sàng tái sử dụng.

## 5. Xây dựng dịch vụ Backend API (FastAPI)
- Khởi tạo thư mục `service_predict`.
- Xây dựng API Server sử dụng **FastAPI** cùng luồng validation dữ liệu bằng **Pydantic**:
  - **Endpoint**: `POST /predict`
  - Nhận Input JSON chứa 7 biến đầu vào kèm `lat` và `long`.
  - Load model `.pkl` để dự đoán và trả về output JSON `predicted_muc_nuoc_t_plus_1` cùng vị trí hiện tại.
- Hỗ trợ file: `requirements.txt`, `Dockerfile`, và code giả lập request `test_api.py`.

## 6. Yêu cầu: Làm lại dữ liệu tách biệt Sông/Hồ
- Cấu hình file `clean_data_split.py` trong thư mục `add_lake_training/`.
- Chuẩn hoá cột `type` ("lake" -> "Lake", "river" -> "River").
- Tách 2 tập dữ liệu riêng biệt:
  - **Lake**: Làm sạch tập trung vào cột `Tỷ lệ dung tích (%)`.
  - **River**: Làm sạch toàn bộ các cột, fill missing values bằng Median (theo nhóm từng Trạm), gạt nhiễu 1%-99%.
- (*Chưa chạy kịch bản Train model trên tập mới*).

## 7. Xây dựng Kịch bản Thuyết trình Cấp cuối kỳ
- Cung cấp lời thoại báo cáo bóc tách các pha Data Mining, Feature Engineering và thuật toán XGBoost chi tiết theo slide. File kịch bản được nháp trong `presentation_script.md`.

---
*Ghi chú: Trợ lý nội bộ không trực tiếp trích xuất được giao diện Chat của hệ thống. Tài liệu này được tạo tự động nhằm tổng hợp mọi hành động, code và tư duy (context) từ đầu cuộc hội thoại đến nay.*
