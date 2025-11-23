# ===================== IMPORTS ===================== #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
from pyvi.ViTokenizer import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import streamlit as st
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from category_encoders import CatBoostEncoder
from PIL import Image
import io

# ==============================================
#  ĐƯỜNG DẪN FILE DỮ LIỆU CHUNG CHO ADMIN + USER
# ==============================================
DATA_PATH = "uploaded_data.xlsx"

# ==============================================
#  KHỞI TẠO TRẠNG THÁI CHUNG
# ==============================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None        # "user" hoặc "admin"
if "file_ready" not in st.session_state:
    st.session_state.file_ready = False     # Đã có file dữ liệu chung chưa

# ===================== CUSTOM CSS ===================== #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Title Style */
    .title-center {
        text-align: center;
        font-size: 40px !important;
        font-weight: 600 !important;
        color: #2C3E50;
        padding-bottom: 10px;
    }

    /* Header Gradient */
    .header {
        background: linear-gradient(90deg, #0062E6, #33AEFF);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }

    /* Card Style */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }

    .card h3 {
        margin: 0;
        color: #2C3E50;
        font-weight: 600;
    }

    .card p {
        margin: 0;
        font-size: 24px;
        color: #2980B9;
        font-weight: 600;
    }

    /* Center Image */
    .center-img {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    /* FULL PAGE WIDTH */
    .block-container {
        max-width: 77% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* FIX COLUMN WIDTH */
    .css-1r6slb0, .css-12oz5g7 {
        flex: 1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ================= SẢNH CHỜ CHỌN CHẾ ĐỘ ================= #
def show_lobby():
    """Sảnh chờ: buộc chọn chế độ trước khi vào app chính"""
    st.title("🚪 Chào mừng bạn đến với hệ thống phân tích xe máy")
    st.write("Vui lòng chọn **chế độ sử dụng** để tiếp tục:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Người dùng", use_container_width=True):
            # Người dùng vào trước khi có dữ liệu -> chỉ xem được 3 trang
            st.session_state.app_mode = "user"
            st.rerun()
    with col2:
        if st.button("🛠 Quản trị", use_container_width=True):
            st.session_state.app_mode = "admin"
            st.rerun()

    st.stop()

# Nếu chưa chọn mode → sảnh chờ
if st.session_state.app_mode is None:
    show_lobby()

# ============================================
#  BACKEND HÀM DUY NHẤT (LOAD + TIỀN XỬ LÝ)
# ============================================
@st.cache_resource
@st.cache_resource
def load_backend(file_content):

    df = pd.read_excel(io.BytesIO(file_content))

    # ==== Load stopwords ====
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STOP_WORD_FILE = os.path.join(BASE_DIR, "vietnamese-stopwords.txt")
    with open(STOP_WORD_FILE, "r", encoding="utf-8") as f:
        stop_words = f.read().split("\n")

    # ==== Load dictionaries ====
    def load_dict(path):
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    d[parts[0]] = parts[1]
        return d

    emoji_dict = load_dict(os.path.join(BASE_DIR, "emojicon.txt"))
    wrong_dict = load_dict(os.path.join(BASE_DIR, "wrong-word.txt"))

    df["id"] = range(len(df))

    # ==== Text utils ====
    def normalize_text_light(text):
        text = str(text).lower()
        for k, v in emoji_dict.items():
            text = text.replace(k, f" {v} ")
        for k, v in wrong_dict.items():
            text = text.replace(k, f" {v} ")
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def remove_stopwords(text):
        return " ".join([w for w in text.split() if w not in stop_words])

    def preprocess_text(text):
        return " ".join(tokenize(remove_stopwords(normalize_text_light(text))))

    # ==== TF-IDF ====
    if "Content_wt_joined" not in df.columns:
        df["Content_wt"] = df["Content"].apply(normalize_text_light).apply(remove_stopwords)
        df["Content_wt_joined"] = df["Content_wt"].apply(lambda x: " ".join(tokenize(x)))

    df.loc[df["Content_wt_joined"].str.strip() == "", "Content_wt_joined"] = df["Tiêu đề"]

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df["Content_wt_joined"])

    # ==== Load cosine_matrix ====
    with open(os.path.join(BASE_DIR, "Cosine_similarity_matrix.pkl"), "rb") as f:
        cosine_sim = pickle.load(f)

    return df, cosine_sim, vectorizer, tfidf_matrix, normalize_text_light, remove_stopwords, preprocess_text

# ===================== SIDEBAR & CHUYỂN MODE ===================== #

st.sidebar.title("⚙ Menu")

# Thông báo chế độ hiện tại
if st.session_state.app_mode == "user":
    st.sidebar.markdown("**Chế độ hiện tại:** 👤 Người dùng")
else:
    st.sidebar.markdown("**Chế độ hiện tại:** 🛠 Quản trị")

uploaded_file = None

# ---- ADMIN: có quyền upload file ----
if st.session_state.app_mode == "admin":
    uploaded_file = st.sidebar.file_uploader("📤 Tải lên file Excel", type=["xlsx", "xls"])
    if uploaded_file is not None:

        # Ghi dữ liệu file vào RAM
        st.session_state["excel_bytes"] = uploaded_file.getvalue()
        st.session_state.file_ready = True

        st.success("✅ Đã tải dữ liệu vào RAM!")

        # Không rerun ở đây, rerun sẽ làm sidebar mất widget
        # st.rerun()

# ---- Widget chuyển mode (chỉ xuất hiện khi đã có dữ liệu) ----
if st.session_state.file_ready:
    with st.sidebar.expander("🔁 Chuyển đổi chế độ"):
        if st.session_state.app_mode == "admin":
            if st.button("👤 Chuyển sang Người dùng", use_container_width=True):
                st.session_state.app_mode = "user"
                st.rerun()
        else:
            if st.button("🛠 Chuyển sang Quản trị", use_container_width=True):
                st.session_state.app_mode = "admin"
                st.rerun()

# ===================== XÂY DỰNG MENU ===================== #

# Người dùng:
#   - nếu chưa có dữ liệu -> 3 trang
#   - nếu đã có dữ liệu do admin upload -> full chức năng (nhưng không được upload)
if st.session_state.app_mode == "user":
    if st.session_state.file_ready:
        menu = [
            "Trang chủ",
            "Mô tả ứng dụng",
            "Bảng điều hướng",
            "Đề xuất & Phân cụm",
            "Trực quan hóa",
            "Phụ trách ứng dụng"
        ]
    else:
        menu = ["Trang chủ", "Mô tả ứng dụng", "Phụ trách ứng dụng"]

# Admin:
#   - trước khi upload -> 3 trang
#   - sau khi upload -> full chức năng
else:
    if st.session_state.file_ready:
        menu = [
            "Trang chủ",
            "Mô tả ứng dụng",
            "Bảng điều hướng",
            "Đề xuất & Phân cụm",
            "Trực quan hóa",
            "Phụ trách ứng dụng"
        ]
    else:
        menu = ["Trang chủ", "Mô tả ứng dụng", "Phụ trách ứng dụng"]

page = st.sidebar.radio("Go to:", menu)

# ===================== LOAD DATA CHUNG (ADMIN + USER) ===================== #
df = None
cosine_sim = vectorizer = tfidf_matrix = None
normalize_text_light = remove_stopwords = preprocess_text = None

df = None
if st.session_state.file_ready and "excel_bytes" in st.session_state:
    df, cosine_sim, vectorizer, tfidf_matrix, normalize_text_light, remove_stopwords, preprocess_text = \
        load_backend(st.session_state["excel_bytes"])


# ===== Khởi tạo giá trị mặc định cho card thống kê =====
if "total_items" not in st.session_state:
    st.session_state["total_items"] = len(df) if df is not None else 0
if "total_clusters" not in st.session_state:
    st.session_state["total_clusters"] = 0

def require_file_loaded():
    """Dùng cho cả Admin + User: buộc phải có dữ liệu trước khi dùng module."""
    if df is None:
        st.warning("⚠ Vui lòng để Admin upload file Excel trước khi sử dụng chức năng này.")
        st.stop()

#############################################################
# =========== RECOMMEND FUNCTIONS =========== #

def get_recomendations(id, cosine_sim=cosine_sim, nums=7):
    idx = df.index[df["id"] == id][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:nums+1]

    indices = [i[0] for i in scores]
    res = df.iloc[indices][["id", "Tiêu đề", "Thương hiệu", "Giá", "Số Km đã đi", "Địa chỉ"]].copy()
    res["Cosine_Similarity"] = [round(i[1], 3) for i in scores]
    return res

def recommend_by_keyword(keyword, nums=7):
    keyword_clean = preprocess_text(keyword)
    if keyword_clean.strip() == "":
        return df.head(nums)

    keyword_vec = vectorizer.transform([keyword_clean])
    scores = cosine_similarity(keyword_vec, tfidf_matrix).flatten()

    if scores.max() == 0:
        return df.head(nums)

    top_idx = scores.argsort()[::-1][:nums]
    res = df.iloc[top_idx][["id", "Tiêu đề", "Thương hiệu", "Giá", "Dung tích xe", "Số Km đã đi", "Địa chỉ"]]
    res["Cosine_Similarity"] = scores[top_idx]
    return res

#############################################################
# ===================== HEADER & CARDS ===================== #
st.markdown("<div class='header'>Bảng điều khiển dự đoán & phân cụm xe máy</div>",
            unsafe_allow_html=True)

st.markdown("<h1 class='title-center'>Ứng dụng dự đoán và phân cụm xe máy</h1>",
            unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "Logo_ChoTot.png")
st.image(image_path, use_container_width=True)

col1, col2 = st.columns([1, 1])
total_items = st.session_state.get("total_items", len(df) if df is not None else 0)
cluster_count = st.session_state.get("total_clusters", 0)

with col1:
    st.markdown(f"""
        <div class='card'>
            <h3>Tổng sản phẩm đã xử lý</h3>
            <p>{total_items}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='card'>
            <h3>Số cụm đã xác định</h3>
            <p>{cluster_count}</p>
        </div>
    """, unsafe_allow_html=True)

#############################################################
# ======================= ROUTING ========================= #

# =============== PAGE: Trang chủ =============== #
if page == "Trang chủ":
    st.subheader("🏍️ Chào mừng bạn đến với ứng dụng phân tích xe máy của chúng tôi!")
    st.write("""
        Đặt vấn đề: Một sàn thương mại điện tử (hoặc website rao vặt xe máy cũ như Chợ Tốt,...) đang gặp 3 vấn đề lớn:
        - Người mua khó tìm đúng xe phù hợp vì số tin đăng lớn nhưng hệ thống trả kết quả không thực sự giống với nhu cầu.
        - Người mua không biết mức giá nào là hợp lý có thể cùng một mẫu xe nhưng giá dao động rất mạnh
        - Người bán không biết nhóm khách hàng nào phù hợp với xe của họ để tối ưu hoá việc tiếp cận khách hàng tiềm năng.
             
        Ứng dụng này cho phép bạn:
        - 🔍 Tìm kiếm xe tương tự bằng Hệ thống đề xuất  
        - 📊 Thực hiện phân cụm dựa vào nhiều thuộc tính  
        - 🎨 Trực quan hóa dữ liệu dễ dàng  
    """)
    st.info("Chọn mục ở thanh bên trái để bắt đầu.")

# =============== PAGE: Mô tả ứng dụng =============== #
elif page == "Mô tả ứng dụng":
    st.subheader("📘 Giới thiệu Ứng dụng")
    st.write("""
        Ứng dụng được xây dựng gồm 2 module chính:

        **1️⃣ Hệ thống đề xuất**
        - Tìm những xe máy giống nhất dựa vào ID, Từ khóa,…  
        - Sử dụng TF-IDF + Cosine Similarity  
        - Cho phép gợi ý theo **ID** hoặc theo **Keyword**

        **2️⃣ Phân cụm**
        - Gom nhóm xe theo giá, hãng, dung tích, năm đăng ký…  
        - Thuật toán hỗ trợ:
            - KMeans
            - Agglomerative
            - Gaussian Mixture Model
        - Giảm chiều: PCA, t-SNE, UMAP

        **3️⃣ Trực quan hóa**
        - Wordcloud
        - Biểu đồ phân bố giá
        - Countplot thương hiệu  
    """)

# =============== PAGE: Bảng điều hướng =============== #
elif page == "Bảng điều hướng":
    require_file_loaded()

    st.subheader("🛠 Control Panel Settings")
    st.write("Cấu hình chung cho app (tuỳ chọn mở rộng):")

    items = st.slider("Số lượng items hiển thị", 5, 50, 10)
    show_price = st.checkbox("Hiển thị thông tin giá", True)
    show_brand = st.checkbox("Hiển thị thương hiệu", True)

    st.success("Cài đặt đã được áp dụng.")

# =============== PAGE: Đề xuất & Phân cụm =============== #
elif page == "Đề xuất & Phân cụm":
    require_file_loaded()

    tab1, tab2 = st.tabs(["🔍 Hệ thống đề xuất", "📦 Phân cụm"])

    # TAB 1 - Đề xuất
    with tab1:
        st.header("🔍 Motorcycle Hệ thống đề xuất")
        rec_mode = st.radio("Chọn cách gợi ý:", ["Theo danh mục có sẵn", "Theo từ khóa"])

        if rec_mode == "Theo danh mục có sẵn":  
            st.subheader("🔍 Tìm kiếm theo danh mục xe")

            # Nhập từ khóa
            keyword = st.text_input("Nhập từ khóa để lọc danh mục:")

            # Tạo danh sách gợi ý danh mục
            if keyword.strip() == "":
                # 10 danh mục ngẫu nhiên
                suggested_titles = df["Tiêu đề"].sample(10, random_state=42).tolist()
            else:
                # Lọc theo keyword (không phân biệt hoa thường)
                suggested_titles = df[df["Tiêu đề"].str.contains(keyword, case=False, na=False)] \
                                    ["Tiêu đề"].head(10).tolist()

                if len(suggested_titles) == 0:
                    st.warning("❗ Không tìm thấy danh mục phù hợp. Hiển thị danh mục ngẫu nhiên.")
                    suggested_titles = df["Tiêu đề"].sample(10, random_state=42).tolist()

            # Chọn tiêu đề
            selected_title = st.selectbox("Chọn danh mục xe cần tìm gợi ý:", suggested_titles)

            # Số lượng gợi ý
            nums = st.slider("Số lượng gợi ý:", 3, 20, 7)

            # Lấy ID từ tiêu đề đã chọn
            selected_id = int(df[df["Tiêu đề"] == selected_title]["id"].values[0])

            if st.button("🔎 Gợi ý theo danh mục"):
                result = get_recomendations(selected_id, nums=nums)
                st.markdown("""
                **🔹 Cosine Similarity** 
                - Giá trị từ **0 → 1**. Càng gần **1** → Hai mô tả xe càng giống nhau.  
                - **> 0.7** → Tương đồng mạnh (rất liên quan).  
                - 0.4 – 0.7 → Tương đồng trung bình.  
                - **< 0.3** → Tương đồng thấp.
                """)                 
                st.dataframe(result)

                # WordCloud từ các tiêu đề gợi ý
                text = " ".join(result["Tiêu đề"].astype(str))
                wc = WordCloud(width=800, height=350, background_color="white").generate(text)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)               

        if rec_mode == "Theo từ khóa":
            keyword = st.text_input("Nhập từ khóa:")
            nums = st.slider("Số lượng gợi ý:", 3, 20, 7)

            if st.button("🔎 Gợi ý theo từ khóa"):
                result = recommend_by_keyword(keyword, nums)
                st.markdown("""
                **🔹 Cosine Similarity** 
                - Giá trị từ **0 → 1**. Càng gần **1** → Hai mô tả xe càng giống nhau.  
                - **> 0.7** → Tương đồng mạnh (rất liên quan).  
                - 0.4 – 0.7 → Tương đồng trung bình.  
                - **< 0.3** → Tương đồng thấp.
                """)                
                st.dataframe(result)

                text = " ".join(result["Tiêu đề"].astype(str))
                wc = WordCloud(width=800, height=350, background_color="white").generate(text)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

    # ---------------- INIT SESSION STATE ----------------
    defaults = {
        "cluster_model": None,
        "labels": None,
        "encoder": None,
        "scaler": None,
        "df2_cluster": None,
        "X2_scaled": None,
        "cluster_summary": None,
        "cluster_labels": {},
        "survey_done": False,
        "inertia": None,
        "sil_scores": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------------- BUILD CLUSTER DATASET ----------------
    def build_cluster_dataset(df):
        """Chuẩn hóa dữ liệu để phân cụm với PowerTransformer"""
        features = ['Giá_num', 'Km_num', 'Dung_tich_num', 'Năm đăng ký', 'Thương hiệu']
        df2 = df[features].dropna()

        encoder = CatBoostEncoder()
        brand_encoded = encoder.fit_transform(df2['Thương hiệu'], df2['Giá_num'])

        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        numeric_scaled = scaler.fit_transform(
            df2[['Giá_num', 'Km_num', 'Dung_tich_num', 'Năm đăng ký']]
        )

        X = np.concatenate([numeric_scaled, brand_encoded.values], axis=1)
        return X, df2, encoder, scaler

    # ---------------- TAB 2: Phân cụm ----------------
    with tab2:
        st.header("📦 Phân cụm xe máy")

        # Build dataset
        X2_scaled, df2_cluster, encoder, scaler = build_cluster_dataset(df)

        # ==============================================
        #   TRƯỜNG HỢP ADMIN — ĐẦY ĐỦ CHỨC NĂNG
        # ==============================================
        if st.session_state.app_mode == "admin":

            # ====== KHẢO SÁT SỐ CỤM ======
            if st.button("🔍 Khảo sát số cụm"):
                K_range = range(2, 10)
                inertia = []
                sil_scores = []

                st.session_state.survey_done = False

                for k_tmp in K_range:
                    kmeans_tmp = KMeans(n_clusters=k_tmp, random_state=42)
                    labels_tmp = kmeans_tmp.fit_predict(X2_scaled)
                    inertia.append(kmeans_tmp.inertia_)
                    sil_scores.append(silhouette_score(X2_scaled, labels_tmp))

                st.session_state.survey_done = True
                st.session_state.inertia = inertia
                st.session_state.sil_scores = sil_scores
                st.success("Đã hoàn thành khảo sát số cụm!")

            if st.session_state.get("survey_done", False):
                K_range = range(2, 10)

                fig_elbow, ax_elbow = plt.subplots()
                ax_elbow.plot(K_range, st.session_state.inertia, "o-")
                ax_elbow.set_xlabel("Số cụm (k)")
                ax_elbow.set_ylabel("Inertia")
                ax_elbow.set_title("Biểu đồ Elbow")
                st.pyplot(fig_elbow)

                fig_sil, ax_sil = plt.subplots()
                ax_sil.plot(K_range, st.session_state.sil_scores, "o-")
                ax_sil.set_xlabel("Số cụm (k)")
                ax_sil.set_ylabel("Silhouette Score")
                ax_sil.set_title("Biểu đồ Silhouette")
                st.pyplot(fig_sil)

            # ===== CHỌN THUẬT TOÁN, CHẠY PHÂN CỤM =====
            k = st.number_input("🔢 Chọn số cụm tối ưu", min_value=2, max_value=15, value=4, step=1)
            algo = st.selectbox(
                "Chọn thuật toán phân cụm",
                ["KMeans", "Gaussian Mixture", "Agglomerative"]
            )

            if st.button("🚀 Chạy phân cụm"):
                if algo == "KMeans":
                    model = KMeans(n_clusters=k, random_state=42)
                    labels = model.fit_predict(X2_scaled)
                elif algo == "Gaussian Mixture":
                    model = GaussianMixture(n_components=k, random_state=42)
                    labels = model.fit_predict(X2_scaled)
                else:
                    model = AgglomerativeClustering(n_clusters=k)
                    labels = model.fit_predict(X2_scaled)

                sil = silhouette_score(X2_scaled, labels)

                st.session_state.cluster_model = model
                st.session_state.labels = labels
                st.session_state.X2_scaled = X2_scaled
                st.session_state.encoder = encoder
                st.session_state.scaler = scaler

                df2_cluster['Cluster'] = labels
                st.session_state.df2_cluster = df2_cluster.copy()

                st.session_state.cluster_labels = {}
                st.session_state.total_items = len(df2_cluster)
                st.session_state.total_clusters = k

                st.success(f"🎯 Đã phân cụm thành công bằng {algo} — Silhouette = {sil:.3f}")
                st.markdown("""
                **🔹 Silhouette Score**
                - Đánh giá **mức độ tách biệt giữa các cụm** và **mức độ tập trung trong từng cụm**. Giá trị nằm trong khoảng **[-1, 1]**.  
                - Càng gần **1** → Cụm phân chia càng rõ ràng, dễ tách biệt.  
                - Từ **0.5 trở lên** → Chất lượng phân cụm tốt.  
                - Từ **0.3 – 0.5** → Chấp nhận được.  
                - Dưới **0.25** → Cụm chồng chéo, chất lượng chưa tốt.
                """)
                # PCA visualization
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X2_scaled)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=labels, palette="tab10", ax=ax)
                st.pyplot(fig)

                # Summary
                cluster_counts = df2_cluster['Cluster'].value_counts().sort_index()
                cluster_means = df2_cluster.groupby('Cluster')[['Giá_num', 'Km_num', 'Dung_tich_num', 'Năm đăng ký']].mean()
                summary = pd.concat([
                    cluster_counts.rename("Số lượng"),
                    cluster_means
                ], axis=1)

                st.session_state.cluster_summary = summary.copy()

            # ====== FORM ĐẶT TÊN CỤM ======
            if st.session_state.cluster_summary is not None:
                st.subheader("✏️ Đặt tên cho từng cụm")

                with st.form("form_cluster_name"):
                    new_labels = {}
                    for cid in st.session_state.cluster_summary.index:
                        default = st.session_state.cluster_labels.get(cid, f"Cụm {cid}")
                        new_labels[cid] = st.text_input(f"Tên cụm {cid}", value=default)
                    submitted = st.form_submit_button("💾 Lưu tên cụm")

                if submitted:
                    st.session_state.cluster_labels = new_labels
                    updated = st.session_state.cluster_summary.copy()
                    updated["Tên cụm"] = [new_labels[c] for c in updated.index]
                    cols = ["Tên cụm"] + [c for c in updated.columns if c != "Tên cụm"]
                    updated = updated[cols]
                    st.session_state.cluster_summary = updated
                    st.success("✔ Đã cập nhật tên cụm!")

        # ==============================================
        #     TRƯỜNG HỢP NGƯỜI DÙNG — CHỈ ĐƯỢC XEM
        # ==============================================
        else:
            st.info("👤 Bạn đang ở chế độ Người dùng — chỉ được xem kết quả phân cụm đã được Admin cấu hình.")

        # ===== HIỂN THỊ BẢNG THỐNG KÊ (DÙ ADMIN HAY USER) =====
        if st.session_state.cluster_summary is not None:
            st.subheader("📊 Bảng thống kê cụm (đã cập nhật)")
            st.dataframe(st.session_state.cluster_summary)

        # ===== DỰ ĐOÁN CỤM CHO XE MỚI (CẢ USER & ADMIN ĐỀU XÀI ĐƯỢC) =====
        st.subheader("🔮 Dự đoán cụm cho xe mới")

        if st.session_state.cluster_model is None:
            st.warning("⚠ Bạn cần để Admin chạy phân cụm trước!")
        else:
            gia = st.number_input("Giá xe", min_value=0)
            km = st.number_input("Km đã đi", min_value=0)
            cc = st.number_input("Dung tích (cc)", min_value=50)
            year = st.number_input("Năm đăng ký", min_value=1990, max_value=2025)
            brand = st.selectbox("Thương hiệu", df['Thương hiệu'].unique())

            if st.button("🔍 Phân cụm xe của bạn"):
                encoder = st.session_state.encoder
                scaler = st.session_state.scaler
                model = st.session_state.cluster_model

                new_brand = encoder.transform(pd.DataFrame({"Thương hiệu": [brand]}))
                new_numeric = scaler.transform([[gia, km, cc, year]])
                X_new = np.concatenate([new_numeric, new_brand.values], axis=1)

                if hasattr(model, "predict"):
                    cluster_id = model.predict(X_new)[0]
                else:
                    centroids = np.vstack([
                        st.session_state.X2_scaled[st.session_state.labels == c].mean(axis=0)
                        for c in range(st.session_state.total_clusters or 4)
                    ])
                    cluster_id = np.argmin(np.linalg.norm(centroids - X_new, axis=1))

                cluster_label = st.session_state.cluster_labels.get(cluster_id, f"Cụm {cluster_id}")
                st.success(f"✔ Xe của bạn thuộc **Cụm {cluster_id} – {cluster_label}!**")

# =============== PAGE: Trực quan hóa =============== #
elif page == "Trực quan hóa":
    require_file_loaded()

    st.subheader("📊 Trực quan hóa Dashboard")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Giá"].dropna(), kde=True, ax=ax1)
    st.pyplot(fig1)

    text = " ".join(df["Tiêu đề"].astype(str))
    wc = WordCloud(width=900, height=400, background_color="white").generate(text)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

    st.subheader("📊 Biểu đồ tần suất thương hiệu")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="Thương hiệu", order=df["Thương hiệu"].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =============== PAGE: Phụ trách ứng dụng =============== #
elif page == "Phụ trách ứng dụng":
    st.subheader("📋 Phụ trách ứng dụng")
    st.markdown("""
        ### 🧑‍💻 Bảng phân công công việc
        
        | Thành viên           | Công việc |
        |----------------------|-----------|
        | **Nguyễn Duy Thanh** | GUI for Hệ thống đề xuất and phân cụm |
        | **Nguyễn Thái Bình** | GUI for Price Prediction and Anomaly Detection |
    """)

#############################################################
# ===================== FOOTER ===================== #

st.sidebar.markdown("---")

avatar1_path = os.path.join(BASE_DIR, "avatar.jpg")
avatar2_path = os.path.join(BASE_DIR, "avatar_2.jpg")

avatar1 = Image.open(avatar1_path)
avatar2 = Image.open(avatar2_path)

def crop_avatar(img, offset_ratio=0.10):
    w, h = img.size
    size = min(w, h)
    offset = int(size * offset_ratio)

    left = (w - size) / 2
    top = offset
    right = (w + size) / 2
    bottom = offset + size
    bottom = min(bottom, h)

    img = img.crop((left, top, right, bottom))
    img = img.resize((80, 80), Image.LANCZOS)
    return img

avatar1 = crop_avatar(avatar1)
avatar2 = crop_avatar(avatar2)

colA, colB = st.sidebar.columns(2)
with colA:
    st.image(avatar1, width=80)
with colB:
    st.image(avatar2, width=80)

st.sidebar.markdown("""
**Designed by:**  
**Nguyễn Duy Thanh**  
Email: [duythanh200620@gmail.com](mailto:duythanh200620@gmail.com)  
**Nguyễn Thái Bình**  
Email: [thaibinh782k1@gmail.com](mailto:thaibinh782k1@gmail.com)
""")
