import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor

# Import các hàm và hằng số từ file xử lý
from du_bao_gia import get_encoded_features, predict_price, MODEL_RESOURCES_PATH

# --- KHỞI TẠO TÀI NGUYÊN (CHỈ CHẠY 1 LẦN NHỜ CACHING) ---
# Tải tất cả các resource (model, encoder, scaler) từ file pickle
@st.cache_resource
def load_all_resources(model_path):
    try:
        with open(model_path, 'rb') as f:
            resources = pickle.load(f)
        return resources
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file tài nguyên mô hình: {model_path}. Vui lòng chạy file 'model_utils.py' trước!")
        return None


st.title("Trung Tâm Tin Học")
# st.image("xe_may_cu.jpg", caption="Xe máy cũ")
st.image("./GUI_XeMayCu_Copy/xe_may_cu.jpg", caption="Xe máy cũ")
# # tạo dataframe mẫu, có 3 cột: Thương hiệu, số lượng xe, Giá trung bình
# data = {
#     'Thương hiệu': ['Honda', 'Yamaha', 'Suzuki', 'Piaggio', 'SYM'],
#     'Số lượng xe': [150, 120, 90, 60, 80],
#     'Giá trung bình (triệu VND)': [15.5, 14.0, 13.5, 16.0, 12.5]
# }
# df = pd.DataFrame(data)
# st.subheader("Dữ liệu xe máy cũ")
# st.dataframe(df)

# # Vẽ biểu đồ số lượng xe theo thương hiệu
# st.subheader("Biểu đồ số lượng xe theo thương hiệu")
# fig, ax = plt.subplots()
# sns.barplot(x='Thương hiệu', y='Số lượng xe', data=df, ax=ax)
# st.pyplot(fig)

# st.image("thong_ke.jpg", caption="Thong ke xe may cu")

menu = ["Home", "Capstone Project", "Sử dụng các điều khiển", "Gợi ý điều khiển project 1", "Gợi ý điều khiển project 2"]

choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")
    st.write('''
            ### Chào mừng các bạn đến với khóa học
            #### Đồ án tốt nghiệp''')
              
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 2 chủ đề trong khóa học:    
    - Topic 1: Dự đoán giá xe máy cũ, phát hiện xe máy bất thường
    - Topic 2: Hệ thống gợi ý xe máy dựa trên nội dung, phân cụm xe máy
             """)

elif choice == 'Sử dụng các điều khiển':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)
    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")
    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")
    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)
    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)
    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)
    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)    
    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)
    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)
    # 11. Display JSON
    st.subheader("11. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)
    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)
          
elif choice == 'Gợi ý điều khiển project 1':
    # 1. Tải tài nguyên (Mô hình, Encoder, Scaler)
    resources = load_all_resources(MODEL_RESOURCES_PATH)
    if resources is None:
        st.stop()

    # Lấy các giá trị duy nhất từ DataFrame đã làm sạch trong resources để điền vào Selectbox
    df_for_encoder = resources['df_for_encoder']
    THUONG_HIEU_LIST = df_for_encoder['Thương hiệu'].unique()
    DONG_XE_LIST = df_for_encoder['Dòng xe'].unique()
    # TINH_TRANG_LIST = df_for_encoder['Tình trạng'].unique()
    LOAI_XE_LIST = df_for_encoder['Loại xe'].unique()
    DUNG_TICH_LIST = df_for_encoder['Dung tích xe'].unique()
    XUAT_XU_LIST = df_for_encoder['Xuất xứ'].unique()  # Danh sách Xuất xứ

    st.write("##### Gợi ý điều khiển project 1: Dự đoán giá xe máy cũ và phát hiện xe máy bất thường")
    st.write("##### Dữ liệu mẫu")
    # đọc dữ liệu từ file subset_100motobykes.csv
    df = pd.read_csv("subset_100motobikes.csv")
    st.dataframe(df.head())

    # Trường hợp 2: Đọc dữ liệu từ file csv, excel do người dùng tải lên
    st.write("### Đọc dữ liệu từ file csv do người dùng tải lên")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        st.write("Dữ liệu đã nhập:")
        st.dataframe(df_up.head())

    st.write("### 1. Dự đoán giá xe máy cũ")

    # Tạo điều khiển để người dùng nhập các thông tin về xe máy
    col1, col2 = st.columns(2)
    with col1:
        thuong_hieu = st.selectbox("Chọn hãng xe", THUONG_HIEU_LIST)
        dong_xe = st.selectbox("Chọn dòng xe", DONG_XE_LIST)
        # tinh_trang = st.selectbox("Chọn tình trạng", TINH_TRANG_LIST)
        loai_xe = st.selectbox("Chọn loại xe", LOAI_XE_LIST)
    with col2:
        dung_tich_xi_lanh = st.selectbox("Dung tích xi lanh (cc)", DUNG_TICH_LIST)
        nam_dang_ky = st.slider("Năm đăng ký", 2000, 2024, 2015)
        so_km_da_di = st.number_input("Số km đã đi", min_value=0, max_value=200000, value=50000, step=1000)
        # Thêm ô Xuất xứ
        xuat_xu = st.selectbox("Xuất xứ", XUAT_XU_LIST)

    du_doan_gia = st.button("Dự đoán giá")

    if du_doan_gia:
        st.write("Hãng xe:", thuong_hieu)
        st.write("Dòng xe:", dong_xe)
        # st.write("Tình trạng:", tinh_trang)
        st.write("Loại xe:", loai_xe)
        st.write("Dung tích xi lanh (cc):", dung_tich_xi_lanh)
        st.write("Năm đăng ký:", nam_dang_ky)
        st.write("Số km đã đi:", so_km_da_di)

        # 2. Mã hóa và chuẩn hóa feature đầu vào
        features_scaled = get_encoded_features(
            thuong_hieu=thuong_hieu,
            dong_xe=dong_xe,
            loai_xe=loai_xe,
            dung_tich_xi_lanh=dung_tich_xi_lanh,
            nam_dang_ky=nam_dang_ky,
            so_km_da_di=so_km_da_di,
            xuat_xu=xuat_xu,  # Truyền Xuất xứ
            resources=resources
        )

        if features_scaled is not None:
            # 3. Dự đoán giá
            gia_du_doan = predict_price(features_scaled, resources)

            st.success(f"**Giá dự đoán:** {gia_du_doan:,.0f} VND ")

    # Làm tiếp cho phần phát hiện xe máy bất thường
    st.write("### 2. Phát hiện xe máy bất thường")
    so_km_bat_thuong = st.number_input("Nhập số km đã đi để kiểm tra bất thường", min_value=0, max_value=200000, value=50000, step=1000)
    gia_du_doan = st.number_input("Nhập giá dự đoán (VND) để kiểm tra bất thường", min_value=0, max_value=100000000, value=15000000, step=100000)
    kiem_tra_bat_thuong = st.button("Kiểm tra bất thường")
    if kiem_tra_bat_thuong:
        # In ra các thông tin đã chọn        
        st.write("Số km đã đi:", so_km_bat_thuong)
        st.write("Giá dự đoán (VND):", gia_du_doan)
        # Giả sử nếu số km đã đi > 150000 hoặc giá dự đoán < 5000000 thì là bất thường
        if so_km_bat_thuong > 150000 or gia_du_doan < 5000000:
            st.write("#### Xe máy bất thường")
        else:
            st.write("#### Xe máy bình thường.")
        # Trên thực tế cần dùng mô hình phát hiện bất thường để kiểm tra
        # Nếu có mô hình ML, có thể gọi hàm dự đoán ở đây
        pass

elif choice=='Gợi ý điều khiển project 2':
    st.write("##### Gợi ý điều khiển project 2: Recommender System")
    st.write("##### Dữ liệu mẫu")
    # Tạo dataframe có 3 cột là id, title, description
    # Đọc dữ liệu từ file mau_xe_may.xlsx
    df = pd.read_excel("mau_xe_may.xlsx")
    df = pd.read_excel("GUI_XeMayCu - Copy/mau_xe_may.xlsx")
    st.dataframe(df)
    st.write("### 1. Tìm kiếm xe tương tự")
    # Tạo điều khiển để người dùng chọn công ty
    selected_bike = st.selectbox("Chọn xe", df['title'])
    st.write("Xe đã chọn:", selected_bike) 
    # Từ xe đã chọn này, người dùng có thể xem thông tin chi tiết của xe
    # hoặc thực hiện các xử lý khác
    # tạo điều khiển để người dùng tìm kiếm xe dựa trên thông tin người dùng nhập
    search = st.text_input("Nhập thông tin tìm kiếm")
    # Tìm kiếm xe dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    # Trên thực tế sử dụng content-based filtering (cosine similarity/ gensim) để tìm kiếm xe tương tự
    result = df[df['title'].str.lower().str.contains(search.lower())]    
    # tạo button submit
    tim_kiem = st.button("Tìm kiếm")
    if tim_kiem:
        st.write("Danh sách xe tìm được:")
        st.dataframe(result)
       
# Done





