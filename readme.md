# 🛡️ Ứng dụng Dự Báo Rủi Rõ & Phát Hiện Giao Dịch Gian Lận

Ứng dụng web được xây dựng bằng framework **Streamlit** giúp chuyển đổi mô hình phân tích và phát hiện gian lận tài chính từ định dạng Jupyter Notebook sang nền tảng tương tác thời gian thực cho người dùng cuối.

## ✨ Tính năng chính
- **Cấu hình động:** Lựa chọn linh hoạt giữa 3 thuật toán: `Random Forest`, `Decision Tree` và `Logistic Regression`. Thay đổi siêu tham số (`n_estimators`, `max_depth`, `max_iter`,...) trực tiếp trên giao diện.
- **Xử lý mất cân bằng tự động:** Tích hợp kỹ thuật **SMOTE** (Synthetic Minority Over-sampling Technique) để xử lý dữ liệu mất cân bằng nghiêm trọng giữa giao dịch thông thường và giao dịch gian lận.
- **Trực quan hóa trực quan:** Theo dõi cấu trúc dữ liệu thô, phân phối tần suất lớp mục tiêu, kiểm tra biến ngoại lai bằng Boxplot và Ma trận tương quan bằng Plotly.
- **Kiểm định minh bạch:** Đánh giá độ chính xác của mô hình qua bộ chỉ số chuẩn (`Accuracy`, `Precision`, `Recall`, `F1-Score`), cùng với đồ thị trực quan `Confusion Matrix` và đường cong `ROC-AUC`.
- **Dự báo đa chế độ:** Hỗ trợ nhập trực tiếp dữ liệu đơn lẻ trên form biểu mẫu hoặc tải file dữ liệu lớn để tính toán/chấm điểm rủi ro hàng loạt (Batch Prediction) và kết xuất tệp đầu ra.

## 📁 Cấu trúc tệp dữ liệu đầu vào bắt buộc
Tệp dữ liệu tải lên (Train hoặc Test) cần ở định dạng `.csv` hoặc `.xlsx` tuân thủ cấu trúc định dạng cột sau:
- **X_1 đến X_14**: Các biến tính năng liên tục hoặc rời rạc biểu thị đặc trưng giao dịch của khách hàng.
- **default**: Biến mục tiêu phân loại nhị phân nhãn số (`0`: Giao dịch bình thường / Hợp lệ, `1`: Giao dịch có dấu hiệu gian lận / Rủi ro).

## 🛠️ Hướng dẫn cài đặt và khởi chạy

**Bước 1:** Đảm bảo hệ thống máy tính của bạn đã cài đặt môi trường Python (Khuyến nghị phiên bản từ `3.9` đến `3.12`).

**Bước 2:** Mở Terminal/Command Prompt tại thư mục chứa mã nguồn và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
