import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import warnings

warnings.filterwarnings('ignore')

# Đường dẫn đến file output chứa các tài nguyên (model, encoder, scaler)
MODEL_RESOURCES_PATH = 'model_resources.pkl'


def load_and_preprocess_data(data_path):
    """Tải và tiền xử lý dữ liệu."""

    # Tải dữ liệu
    df = pd.read_excel(data_path, index_col=0)

    cols_of_interest = ['Giá', 'Khoảng giá min', 'Khoảng giá max', 'Thương hiệu', 'Dòng xe', 'Năm đăng ký',
                        'Số Km đã đi', 'Loại xe', 'Dung tích xe', 'Xuất xứ']
    df = df[cols_of_interest].copy()

    # Chuyển đổi kiểu dữ liệu
    df['Giá'] = (df['Giá'].str.replace(" đ", "", regex=False).str.replace(".", "", regex=False).astype(float)) / 1000000
    df['Khoảng giá min'] = (df['Khoảng giá min'].str.replace(" tr", "", regex=False).astype(float))
    df['Khoảng giá max'] = (df['Khoảng giá max'].str.replace(" tr", "", regex=False).astype(float))

    df['nam'] = pd.to_numeric(df['Năm đăng ký'], errors='coerce')
    df['nam'] = df['nam'].fillna(df['nam'].median())

    # Loại bỏ null và duplicates
    df.dropna(inplace=True)
    df = df.drop_duplicates()

    # Xóa dòng dữ liệu có Dung tích xe không hợp lệ
    invalid_dt = ['Không biết rõ', 'Đang cập nhật', 'Nhật Bản']
    df = df[~df['Dung tích xe'].isin(invalid_dt)]

    # Xử lý Outlier cột log(Giá)
    df['log_gia'] = np.log1p(df['Giá'])
    mean_val = df['log_gia'].mean()
    stddev_val = df['log_gia'].std()
    low_bound = mean_val - (3 * stddev_val)
    high_bound = mean_val + (3 * stddev_val)
    df = df[(df["log_gia"] < high_bound) & (df["log_gia"] > low_bound)]

    return df


def train_and_save_model(data_path, model_output_path=MODEL_RESOURCES_PATH):
    """
    Tiền xử lý, huấn luyện mô hình XGBoost, và lưu mô hình, encoder, scaler.
    """
    df = load_and_preprocess_data(data_path)

    label_encoders = {}
    scaler = StandardScaler()

    # Mã hóa Label Encoding
    categorical_cols = ['Thương hiệu', 'Dòng xe', 'Loại xe', 'Dung tích xe', 'Xuất xứ']
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col.lower().replace(" ", "")}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le  # Lưu lại bộ mã hóa

    # Chuẩn bị dữ liệu cho mô hình
    feature_cols = ['Số Km đã đi', 'thươnghiệu_encoded', 'dòngxe_encoded',
                    'loạixe_encoded', 'dungtíchxe_encoded', 'xuấtxứ_encoded', 'nam']

    X = df[feature_cols].values
    y = df['log_gia']

    # Chuẩn hóa dữ liệu (Fit scaler trên toàn bộ tập dữ liệu)
    X_scaled = scaler.fit_transform(X)

    # Chia tập huấn luyện/kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình XGBoost
    xgboost_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    xgboost_model.fit(X_train, y_train)

    # Lưu TẤT CẢ TÀI NGUYÊN vào một dictionary
    resources = {
        'model': xgboost_model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'df_for_encoder': df  # Giữ lại DataFrame để biết các giá trị hợp lệ
    }

    # Lưu tài nguyên
    with open(model_output_path, 'wb') as f:
        pickle.dump(resources, f)

    print(f"Tài nguyên mô hình (model, encoder, scaler) đã được lưu tại: {model_output_path}")
    return resources


def get_encoded_features(thuong_hieu, dong_xe, loai_xe, dung_tich_xi_lanh, nam_dang_ky, so_km_da_di, xuat_xu,
                         resources):
    """
    Mã hóa và Chuẩn hóa các feature nhập vào sử dụng các tài nguyên đã được load.
    """

    label_encoders = resources['label_encoders']
    scaler = resources['scaler']

    # Danh sách các biến phân loại cần mã hóa
    categorical_inputs = {
        'Thương hiệu': thuong_hieu,
        'Dòng xe': dong_xe,
        'Loại xe': loai_xe,
        'Dung tích xe': dung_tich_xi_lanh,
        'Xuất xứ': xuat_xu  # Thêm Xuất xứ vào đây
    }

    encoded_features = []

    # 1. Mã hóa Label Encoding cho các biến phân loại
    for col_name, input_value in categorical_inputs.items():
        try:
            le = label_encoders[col_name]
            encoded_value = le.transform([input_value])[0]
            encoded_features.append(encoded_value)
        except ValueError:
            # st.error sẽ được dùng trong file Streamlit
            print(f"Lỗi mã hóa: Giá trị '{input_value}' của '{col_name}' không nằm trong tập dữ liệu huấn luyện.")
            return None

    # 2. Gộp các feature (cần đảm bảo thứ tự khớp với khi huấn luyện mô hình)
    # Thứ tự feature: ['Số Km đã đi', 'thươnghiệu_encoded', 'dòngxe_encoded', 'loạixe_encoded',
    # 'dungtíchxe_encoded', 'xuấtxứ_encoded', 'nam']
    features_list = [
        so_km_da_di,
        *encoded_features,
        nam_dang_ky
    ]

    features_array = np.array(features_list).reshape(1, -1)

    # 3. Chuẩn hóa dữ liệu đầu vào (StandardScaler)
    try:
        features_scaled = scaler.transform(features_array)
        return features_scaled
    except Exception as e:
        print(f"Lỗi khi chuẩn hóa dữ liệu: {e}")
        return None


def predict_price(features_scaled, resources):
    """Dự đoán giá (log-scale) và chuyển đổi ngược về VND."""
    model = resources['model']
    # Dự đoán giá (log-scale)
    log_gia_du_doan = model.predict(features_scaled)[0]

    # Chuyển đổi ngược về đơn vị tiền (VND)
    gia_du_doan = np.exp(log_gia_du_doan)  # exp(x) - 1

    return gia_du_doan


# --- CHẠY TẠO FILE PICKLE (CHẠY LẦN ĐẦU) ---
if __name__ == '__main__':
    try:
        train_and_save_model(data_path='data_motobikes.xlsx', model_output_path=MODEL_RESOURCES_PATH)
    except FileNotFoundError:
        print("ERROR: KHÔNG TÌM THẤY FILE 'mau_xe_may.xlsx'. Vui lòng kiểm tra lại đường dẫn.")