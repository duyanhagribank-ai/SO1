import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# 1. CẤU HÌNH TRANG STREAMLIT ĐẦU TIÊN
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# 2. HÀM NẠP DỮ LIỆU DÙNG CHUNG CÓ CACHE
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# 3. SIDEBAR - VÙNG CẤU HÌNH
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file dữ liệu chứa các tính năng từ X_1 đến X_14 và cột mục tiêu 'default'"
    )
    
    st.divider()
    
    # Lựa chọn mô hình (Notebook dùng 3 mô hình)
    model_option = st.selectbox(
        "Lựa chọn Thuật toán",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        help="Chọn thuật toán Machine Learning muốn huấn luyện."
    )
    
    st.subheader("⚙️ Tham số mô hình AI")
    
    # Hiển thị tham số động dựa theo thuật toán được chọn từ Notebook
    params = {}
    if model_option == "Random Forest":
        params['n_estimators'] = st.slider("n_estimators (Số cây)", min_value=10, max_value=200, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        params['max_depth'] = st.slider("max_depth (Độ sâu tối đa)", min_value=1, max_value=30, value=10, help="Độ sâu tối đa của mỗi cây quyết định.")
        params['random_state'] = st.number_input("random_state", value=42, step=1, help="Trạng thái ngẫu nhiên để tái lặp kết quả.")
    
    elif model_option == "Decision Tree":
        params['criterion'] = st.selectbox("criterion (Tiêu chí phân tách)", options=["gini", "entropy", "log_loss"], index=0)
        params['max_depth'] = st.slider("max_depth (Độ sâu tối đa)", min_value=1, max_value=30, value=5, help="Độ sâu tối đa của cây.")
        params['random_state'] = st.number_input("random_state", value=42, step=1)
        
    elif model_option == "Logistic Regression":
        params['max_iter'] = st.slider("max_iter (Số vòng lặp tối đa)", min_value=100, max_value=2000, value=1000, step=100, help="Số vòng lặp tối đa cho bộ tối ưu hội tụ.")
        params['C'] = st.slider("C (Hệ số nghịch đảo chính quy hóa)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
        params['random_state'] = st.number_input("random_state", value=42, step=1)

    # Nút hành động duy nhất để kích hoạt huấn luyện
    st.divider()
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# 4. HEADER - VÙNG ĐỊNH HƯỚNG
st.title("🛡️ Hệ Thống Dự Báo Rủi Rõ & Phát Hiện Giao Dịch Gian Lận")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu tài chính, tự động xử lý mất cân bằng lớp bằng SMOTE và dự báo khả năng gian lận (mục tiêu: default).")

if uploaded_file is None:
    st.info("💡 Vui lòng tải file dữ liệu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu cấu hình hệ thống.")
    st.stop()

# Đọc file dữ liệu đã upload thông qua hàm cache
file_bytes = uploaded_file.getvalue()
df_main = load_data(file_bytes, uploaded_file.name)

if df_main is None:
    st.error("❌ Định dạng file không hợp lệ hoặc dữ liệu trống.")
    st.stop()

st.caption(f"📁 Đang dùng tệp dữ liệu: `{uploaded_file.name}` | Tổng số dòng: {df_main.shape[0]} | Tổng số cột: {df_main.shape[1]}")
st.divider()

# Thiết lập các biến đặc trưng dựa theo dữ liệu thực tế & Notebook
features = [f"X_{i}" for i in range(1, 15)]
target = "default"

# Kiểm tra sự tồn tại của các cột bắt buộc
missing_cols = [col for col in features + [target] if col not in df_main.columns]
if missing_cols:
    st.error(f"❌ File dữ liệu thiếu các cột bắt buộc sau: {missing_cols}")
    st.stop()

# 5. KHỐI HUẤN LUYỆN (Chỉ chạy khi bấm nút và lưu vào session_state)
if train_clicked:
    with st.spinner("⏳ Đang xử lý dữ liệu và huấn luyện mô hình..."):
        X = df_main[features]
        y = df_main[target]
        
        # Phân chia tập Train/Test theo tỷ lệ của Notebook (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Áp dụng SMOTE để xử lý mất cân bằng lớp giống Notebook
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Khởi tạo mô hình theo lựa chọn
        if model_option == "Random Forest":
            model = RandomForestClassifier(**params)
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(**params)
        else:
            model = LogisticRegression(**params)
            
        # Fit mô hình
        model.fit(X_train_res, y_train_res)
        
        # Dự đoán để lấy chỉ số đánh giá
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu trữ vào session_state 3 thành phần cốt lõi
        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_option
        st.session_state['eval_results'] = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_probs': y_probs,
            'features': features
        }
    st.success(f"🎉 Huấn luyện thành công mô hình {model_option}!")

# 6. GIAO DIỆN CHÍNH - PHÂN VÙNG QUA TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    st.subheader("📋 Phân tích dữ liệu thô và Thống kê mô tả")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số lượng bản ghi (Dòng)", f"{df_main.shape[0]:,}")
    col_m2.metric("Số lượng tính năng (Cột đầu vào)", len(features))
    col_m3.metric("Kích thước tệp", f"{len(file_bytes)/(1024*1024):.2f} MB")
    
    st.write("##### 🕵️ 5 Bản ghi dữ liệu đầu tiên:")
    st.dataframe(df_main.head(), use_container_width=True)
    
    st.write("##### 📉 Thống kê mô tả các biến đặc trưng (X và y):")
    st.dataframe(df_main[features + [target]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("📊 Trực quan hóa phân phối biến và tương quan")
    
    # Lưới 2x2 cho biểu đồ
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    
    # 1. Phân phối biến mục tiêu
    with c1:
        target_counts = df_main[target].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: "0 (Hợp lệ)", 1: "1 (Gian lận)"})
        fig1 = px.bar(target_counts, x='Trạng thái', y='Số lượng', color='Trạng thái',
                     title="Phân phối Biến Mục Tiêu (default)", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig1, use_container_width=True)
        
    # 2. Phân phối của biến X_1 (Biến liên tục tiêu biểu)
    with c2:
        fig2 = px.histogram(df_main, x="X_1", color=target, barmode="overlay",
                            title="Phân phối Tính Năng X_1 theo Lớp Mục Tiêu", color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig2, use_container_width=True)
        
    # 3. Phân phối của biến X_6
    with c3:
        fig3 = px.box(df_main, x=target, y="X_6", color=target,
                      title="Biểu đồ Hộp tính năng X_6 phát hiện ngoại lai", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)
        
    # 4. Biểu đồ tương quan nhiệt giữa các tính năng chính
    with c4:
        corr_matrix = df_main[features[:7] + [target]].corr()
        fig4 = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         title="Ma trận Tương quan Nhiệt (7 biến đầu tiên & target)", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig4, use_container_width=True)

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ---
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng mô hình trên Tập kiểm thử (Test Set)")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Chưa có dữ liệu huấn luyện. Vui lòng bấm nút **'Huấn luyện mô hình'** ở thanh cấu hình bên trái.")
    else:
        model = st.session_state['trained_model']
        res = st.session_state['eval_results']
        
        # Tính toán ma trận nhầm lẫn
        cm = confusion_matrix(res['y_test'], res['y_pred'])
        report_dict = classification_report(res['y_test'], res['y_pred'], output_dict=True)
        
        # Hiển thị các chỉ số chính dưới dạng Metric
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Accuracy (Độ chính xác tổng)", f"{report_dict['accuracy']:.4f}")
        col_r2.metric("Precision (Lớp 1 - Gian lận)", f"{report_dict['1']['precision']:.4f}")
        col_r3.metric("Recall (Lớp 1 - Gian lận)", f"{report_dict['1']['recall']:.4f}")
        col_r4.metric("F1-Score (Lớp 1)", f"{report_dict['1']['f1-score']:.4f}")
        
        st.divider()
        
        col_v1, col_v2 = st.columns(2)
        
        # Trực quan Ma trận nhầm lẫn bằng Plotly Heatmap
        with col_v1:
            st.write("##### 🧮 Ma trận nhầm lẫn (Confusion Matrix):")
            z = cm
            x = ['Dự báo Hợp lệ (0)', 'Dự báo Gian lận (1)']
            y = ['Thực tế Hợp lệ (0)', 'Thực tế Gian lận (1)']
            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
            fig_cm.update_layout(title_text=f"Confusion Matrix - {st.session_state['model_name']}")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        # Trực quan Đường cong ROC và tính diện tích AUC
        with col_v2:
            st.write("##### 📉 Đường cong ROC (Receiver Operating Characteristic):")
            if res['y_probs'] is not None:
                fpr, tpr, thresholds = roc_curve(res['y_test'], res['y_probs'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.area(
                    x=fpr, y=tpr,
                    title=f"Đường cong ROC AUC (Diện tích = {roc_auc:.4f})",
                    labels=dict(x='Tỷ lệ Dương tính giả (FPR)', y='Tỷ lệ Dương tính thật (TPR)'),
                    width=700, height=500
                )
                fig_roc.add_shape(type='line', line=dict(dash='dash', color='red'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("Mô hình được chọn không hỗ trợ xuất xác suất xác thực (predict_proba).")
                
        # Chi tiết Bảng báo cáo phân loại (Classification Report)
        st.write("##### 📄 Báo cáo phân loại chi tiết (Classification Report):")
        df_report = pd.DataFrame(report_dict).transpose()
        st.dataframe(df_report.style.format(precision=4), use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH (DỰ BÁO) ---
with tab4:
    st.subheader("🔮 Hệ thống nhận diện & chấm điểm rủi ro trực tuyến")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Hệ thống dự báo cần mô hình đã được khớp. Vui lòng nhấn nút **'Huấn luyện mô hình'**.")
    else:
        model = st.session_state['trained_model']
        features_list = st.session_state['eval_results']['features']
        
        predict_mode = st.radio(
            "Phương thức nhập dữ liệu đầu vào:",
            options=["Nhập trực tiếp qua Form đơn lẻ", "Tải tệp danh sách hàng loạt (Batch Prediction)"],
            horizontal=True
        )
        
        # Chế độ 1: Nhập trực tiếp qua Form
        if predict_mode == "Nhập trực tiếp qua Form đơn lẻ":
            st.write("👉 Vui lòng điều chỉnh thông số giao dịch bên dưới:")
            
            # Khởi tạo form nhập liệu động cho 14 biến
            with st.form("single_prediction_form"):
                input_data = {}
                # Chia form thành 4 cột cho gọn giao diện
                cols_form = st.columns(4)
                
                for idx, feat in enumerate(features_list):
                    col_idx = idx % 4
                    # Lấy giá trị mặc định dựa vào median của tập dữ liệu chính để tránh người dùng điền sai khoảng
                    default_val = float(df_main[feat].median())
                    min_val = float(df_main[feat].min())
                    max_val = float(df_main[feat].max())
                    
                    with cols_form[col_idx]:
                        input_data[feat] = st.number_input(
                            f"Thông số {feat}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            format="%.4f"
                        )
                
                submit_pred = st.form_submit_button("🔍 Phân tích Rủi ro", type="primary")
                
            if submit_pred:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame đúng định dạng cấu trúc cột lúc Train
                df_input = pd.DataFrame([input_data])
                
                # Tiến hành dự báo rủi ro
                pred_class = model.predict(df_input)[0]
                
                st.markdown("### 📊 Kết quả đánh giá từ AI:")
                if pred_class == 1:
                    st.error("🚨 CẢNH BÁO: Giao dịch này có dấu hiệu GIAN LẬN hoặc RỦI RO CAO!")
                else:
                    st.success("✅ AN TOÀN: Giao dịch được thẩm định là HỢP LỆ.")
                    
                if hasattr(model, "predict_proba"):
                    pred_prob = model.predict_proba(df_input)[0]
                    st.metric("Xác suất lớp rủi ro (Gian lận)", f"{pred_prob[1]*100:.2f}%")
                    st.progress(float(pred_prob[1]))
                    
        # Chế độ 2: Upload file hàng loạt
        else:
            st.write("👉 Hãy tải lên file chứa cấu trúc các cột tính năng đầu vào từ `X_1` đến `X_14`.")
            batch_file = st.file_uploader("Tải tệp dự báo hàng loạt (.csv, .xlsx)", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra xem file batch có đủ 14 biến không
                    missing_batch_cols = [c for c in features_list if c not in df_batch.columns]
                    
                    if missing_batch_cols:
                        st.error(f"❌ File tải lên không hợp lệ, thiếu các cột sau: {missing_batch_cols}")
                    else:
                        # Đảm bảo lấy đúng thứ tự các cột
                        X_batch = df_batch[features_list]
                        
                        # Thực hiện dự báo hàng loạt
                        batch_preds = model.predict(X_batch)
                        
                        df_res = df_batch.copy()
                        df_res['Dự_Báo_Kết_Quả'] = batch_preds
                        df_res['Nhãn_Ý_Nghĩa'] = df_res['Dự_Báo_Kết_Quả'].map({0: "Hợp lệ", 1: "Gian lận/Rủi ro"})
                        
                        if hasattr(model, "predict_proba"):
                            batch_probs = model.predict_proba(X_batch)[:, 1]
                            df_res['Xác_Suất_Rủi_Rõ'] = batch_probs
                        
                        # Tổng kết kết quả dự báo hàng loạt
                        st.success("✅ Đã xử lý và dự báo thành công toàn bộ danh sách!")
                        
                        num_fraud = int((batch_preds == 1).sum())
                        st.metric("Số lượng giao dịch rủi ro phát hiện", f"{num_fraud} / {len(df_batch)}")
                        
                        st.write("##### 📋 Bảng kết quả tổng hợp:")
                        st.dataframe(df_res, use_container_width=True)
                        
                        # Cho phép người dùng tải kết quả về dưới dạng CSV
                        csv_buffer = io.StringIO()
                        df_res.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả (.CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
