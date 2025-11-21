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
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from category_encoders import CatBoostEncoder
from sklearn.manifold import TSNE
from PIL import Image
#############################################################
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

#############################################################
# ============================================
#  BACKEND HÀM DUY NHẤT (TỐI ƯU TOÀN BỘ)
# ============================================

@st.cache_resource
def load_backend():
    import re
    from pyvi.ViTokenizer import tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle

    # ============================================================
    # DEFINE load_dict BEFORE USING IT  ❗❗❗
    # ============================================================
    def load_dict(path):
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    d[parts[0]] = parts[1]
        return d

    # ============================================================
    # BASE DIRECTORY
    # ============================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==== Load stopwords ====
    STOP_WORD_FILE = os.path.join(BASE_DIR, "vietnamese-stopwords.txt")
    with open(STOP_WORD_FILE, "r", encoding="utf-8") as f:
        stop_words = f.read().split("\n")

    # ==== Load dictionaries ====
    emoji_dict = load_dict(os.path.join(BASE_DIR, "emojicon.txt"))
    wrong_dict = load_dict(os.path.join(BASE_DIR, "wrong-word.txt"))

    # ============================================================
    # ==== Normalize text ====
    def normalize_text_light(text):
        text = str(text).lower()
        for k, v in emoji_dict.items():
            text = text.replace(k, f" {v} ")
        for k, v in wrong_dict.items():
            text = text.replace(k, f" {v} ")
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ==== Remove stopwords ====
    def remove_stopwords(text):
        return " ".join([w for w in text.split() if w not in stop_words])

    # ==== Final preprocess ====
    def preprocess_text(text):
        text = normalize_text_light(text)
        text = remove_stopwords(text)
        return " ".join(tokenize(text))

    # ============================================
    #  Load Cleaned Data
    # ============================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ---- Load main dataset (fix path!) ----
    df = pd.read_excel(os.path.join(BASE_DIR, "PJ3_Data_motobikes_cleaned.xlsx"))
    df["id"] = range(len(df))


    # ============================================
    #  Ensure Content_wt_joined exists
    # ============================================
    if "Content_wt_joined" not in df.columns:
        df["Content_wt"] = df["Content"].apply(normalize_text_light).apply(remove_stopwords)
        df["Content_wt_joined"] = df["Content_wt"].apply(lambda x: " ".join(tokenize(x)))

    # Thay chuỗi rỗng
    df.loc[df["Content_wt_joined"].str.strip() == "", "Content_wt_joined"] = df["Tiêu đề"]

    # ============================================
    #  Load cosine similarity
    # ============================================
    with open(r"Cosine_similarity_matrix.pkl", "rb") as f:
        cosine_sim = pickle.load(f)

    # ============================================
    #  Build TF-IDF
    # ============================================
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(df["Content_wt_joined"])

    # ============================================
    #  Return EVERYTHING
    # ============================================
    return (
        df,
        cosine_sim,
        vectorizer,
        tfidf_matrix,
        normalize_text_light,
        remove_stopwords,
        preprocess_text
    )


# ============================
#  GỌI HÀM BACKEND 1 LẦN DUY NHẤT
# ============================
df, cosine_sim, vectorizer, tfidf_matrix, normalize_text_light, remove_stopwords, preprocess_text = load_backend()



#############################################################
# =========== RECOMMEND FUNCTIONS =========== #

def get_recommendations(id, cosine_sim=cosine_sim, nums=7):
    idx = df.index[df["id"] == id][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:nums+1]

    indices = [i[0] for i in scores]
    res = df.iloc[indices][["id","Tiêu đề","Thương hiệu","Giá","Số Km đã đi","Địa chỉ"]].copy()
    res["Cosine_Similarity"] = [round(i[1],3) for i in scores]
    return res

def recommend_by_keyword(keyword, nums=7):
    keyword_clean = preprocess_text(keyword)
    if keyword_clean.strip()=="":
        return df.head(nums)

    keyword_vec = vectorizer.transform([keyword_clean])
    scores = cosine_similarity(keyword_vec, tfidf_matrix).flatten()

    if scores.max()==0:
        return df.head(nums)

    top_idx = scores.argsort()[::-1][:nums]
    res = df.iloc[top_idx][["id","Tiêu đề","Thương hiệu","Giá","Dung tích xe","Số Km đã đi","Địa chỉ"]]
    res["Cosine_Similarity"] = scores[top_idx]
    return res

#############################################################
# ===================== HEADER ===================== #
st.markdown("<div class='header'>Motorcycle Recommendation & Clustering Dashboard</div>",
            unsafe_allow_html=True)

# ===================== TITLE ===================== #
st.markdown("<h1 class='title-center'>An App with a Recommendation System and Clustering</h1>",
            unsafe_allow_html=True)

# ===================== IMAGE CENTER ===================== #
# st.image("Project_3/GUI_XeMayCu/xe_may_cu.jpg", width=450)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== IMAGE CENTER ===================== #
image_path = os.path.join(BASE_DIR, "Logo_ChoTot.png")
st.image(image_path, width="stretch")

# ===================== 2 COLUMNS LAYOUT ===================== #
col1, col2 = st.columns([1, 1])

# Tính tổng số dữ liệu
total_items = len(df)

# Tính số cụm (nếu đã clustering)
if st.session_state.get("labels") is not None:
    cluster_count = len(np.unique(st.session_state.labels))
else:
    cluster_count = 0    # hoặc đặt =4 nếu bạn dùng số cụm cố định

# CARD 1: Total Items
with col1:
    st.markdown(f"""
        <div class='card'>
            <h3>Total Items Processed</h3>
            <p>{total_items}</p>
        </div>
    """, unsafe_allow_html=True)

# CARD 2: Clusters Identified
with col2:
    st.markdown(f"""
        <div class='card'>
            <h3>Clusters Identified</h3>
            <p>4</p>
        </div>
    """, unsafe_allow_html=True)



# ===================== SIDEBAR ===================== #
st.sidebar.title("⚙ Menu")

menu = [
    "Home",
    "App Description",
    "Control panels",
    "Recommendation & Clustering",
    "Visualization",
    "Task assignment"
]

page = st.sidebar.radio("Go to:", menu)

# ======================= ROUTING ========================= #

# =============== PAGE: HOME =============== #
if page == "Home":
    st.subheader("🏍️ Welcome to the Motorcycle Analytics Dashboard")
    st.write("""
        Bussiness Problem: Một sàn thương mại điện tử (hoặc website rao vặt xe máy cũ như Chợ Tốt,...) đang gặp 3 vấn đề lớn:
        - Người mua khó tìm đúng xe phù hợp vì số tin đăng lớn nhưng hệ thống trả kết quả không thực sự giống với nhu cầu.
        - Người mua không biết mức giá nào là hợp lý có thể cùng một mẫu xe nhưng giá dao động rất mạnh
        - Người bán không biết nhóm khách hàng nào phù hợp với xe của họ để tối ưu hoá việc tiếp cận khách hàng tiềm năng.
             
        Ứng dụng này cho phép bạn:
        - 🔍 Tìm kiếm xe tương tự bằng Recommendation System  
        - 📊 Thực hiện phân cụm dựa vào nhiều thuộc tính  
        - 🎨 Trực quan hóa dữ liệu dễ dàng  
    """)
    st.info("Chọn mục ở thanh bên trái để bắt đầu.")

# =============== PAGE: APP DESCRIPTION =============== #
elif page == "App Description":
    st.subheader("📘 Giới thiệu Ứng dụng")
    st.write("""
        Ứng dụng được xây dựng gồm 2 module chính:

        **1️⃣ Recommendation System**
        - Tìm những xe máy giống nhất dựa vào ID, Từ khóa,…
        - Sử dụng TF-IDF + Cosine Similarity  
        - Cho phép gợi ý theo **ID** hoặc theo **Keyword**

        **2️⃣ Clustering**
        - Gom nhóm xe theo giá, hãng, dung tích, năm đăng ký…
        - Thuật toán hỗ trợ:
            - KMeans
            - Agglomerative
            - Gaussian Mixture Model
        - Giảm chiều: PCA, t-SNE, UMAP

        **3️⃣ Visualization**
        - Wordcloud
        - Biểu đồ phân bố giá
        - Countplot thương hiệu  
    """)

# =============== PAGE: CONTROL PANELS =============== #
elif page == "Control panels":
    st.subheader("🛠 Control Panel Settings")
    st.write("Cấu hình chung cho app (tuỳ chọn mở rộng):")

    items = st.slider("Số lượng items hiển thị", 5, 50, 10)
    show_price = st.checkbox("Hiển thị thông tin giá", True)
    show_brand = st.checkbox("Hiển thị thương hiệu", True)

    st.success("Cài đặt đã được áp dụng.")

# =============== PAGE: RECOMMENDATION =============== #
elif page == "Recommendation & Clustering":
    tab1, tab2 = st.tabs(["🔍 Recommendation System", "📦 Clustering"])

    # TAB 1 - Recommendation
    with tab1:
        st.header("🔍 Motorcycle Recommendation System")
        mode = st.radio("Chọn cách gợi ý:", ["Theo ID", "Theo Keyword"])

        # By ID
        if mode=="Theo ID":
            input_id = st.number_input("Nhập ID xe:", min_value=0, max_value=len(df)-1, step=1)
            nums = st.slider("Số lượng gợi ý:", 3,20,7)

            if st.button("🔎 Recommend by ID"):
                result = get_recommendations(int(input_id), nums=nums)
                st.dataframe(result)

                # Wordcloud FIXED
                text = " ".join(result["Tiêu đề"].astype(str))
                wc = WordCloud(width=800, height=350, background_color="white").generate(text)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        # By Keyword
        if mode=="Theo Keyword":
            keyword = st.text_input("Nhập từ khóa:")
            nums = st.slider("Số lượng gợi ý:", 3,20,7)

            if st.button("🔎 Recommend by Keyword"):
                result = recommend_by_keyword(keyword, nums)
                st.dataframe(result)

                # Wordcloud FIXED
                text = " ".join(result["Tiêu đề"].astype(str))
                wc = WordCloud(width=800, height=350, background_color="white").generate(text)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
    
    # ---------------- NAMES FOR CLUSTERS ----------------
    cluster_names = {
        0: "Xe cao cấp",
        1: "Xe phổ thông",
        2: "Xe tầm trung",
        3: "Xe giá rẻ"
    }

    # ---------------- INIT SESSION STATE ----------------
    for key in ["cluster_model", "labels", "encoder", "scaler", "df2_cluster", "X2_scaled"]:
        if key not in st.session_state:
            st.session_state[key] = None


    # ---------------- BUILD CLUSTER DATASET ----------------
    def build_cluster_dataset(df):
        """Chuẩn hóa dữ liệu để clustering với PowerTransformer"""
        features = ['Giá_num', 'Km_num', 'Dung_tich_num', 'Năm đăng ký', 'Thương hiệu']
        df2 = df[features].dropna()

        # 1. Encode thương hiệu bằng CatBoostEncoder
        encoder = CatBoostEncoder()
        brand_encoded = encoder.fit_transform(df2['Thương hiệu'], df2['Giá_num'])

        # 2. Scale numeric bằng PowerTransformer
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        numeric_scaled = scaler.fit_transform(
            df2[['Giá_num', 'Km_num', 'Dung_tich_num', 'Năm đăng ký']]
        )

        # 3. Combine numeric + brand encoding
        X = np.concatenate([numeric_scaled, brand_encoded.values], axis=1)

        return X, df2, encoder, scaler


    # ---------------- TAB 2: CLUSTERING ----------------
    with tab2:

        st.header("📦 Motorcycle Clustering cho bộ dữ liệu này được chia làm 4 cụm")

        algo = st.selectbox(
            "Chọn thuật toán clustering",
            ["KMeans", "Gaussian Mixture", "Agglomerative"]
        )

        # ---------- RUN CLUSTERING ----------
        if st.button("🚀 Chạy clustering"):
            X2_scaled, df2_cluster, encoder, scaler = build_cluster_dataset(df)

            # Chọn mô hình
            if algo == "KMeans":
                model = KMeans(n_clusters=4, random_state=42)
                labels = model.fit_predict(X2_scaled)

            elif algo == "Gaussian Mixture":
                model = GaussianMixture(n_components=4, random_state=42)
                labels = model.fit_predict(X2_scaled)

            else:
                model = AgglomerativeClustering(n_clusters=4)
                labels = model.fit_predict(X2_scaled)

            # Silhouette Score
            sil = silhouette_score(X2_scaled, labels)

            # Lưu session_state
            st.session_state.cluster_model = model
            st.session_state.labels = labels
            st.session_state.X2_scaled = X2_scaled
            st.session_state.encoder = encoder
            st.session_state.scaler = scaler

            df2_cluster['Cluster'] = labels
            st.session_state.df2_cluster = df2_cluster

            st.success(f"🎯 Đã phân cụm thành công bằng {algo} — **Silhouette Score = {sil:.3f}**")
            st.dataframe(df2_cluster.head(20))

            # PCA Plot
            pca = PCA(n_components=2)
            comps = pca.fit_transform(X2_scaled)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=labels, palette="tab10", ax=ax)
            st.pyplot(fig)

        # --------- CURRENT RESULTS ---------
        if st.session_state.cluster_model is not None:
            st.subheader("📌 Kết quả clustering hiện tại")
            st.dataframe(st.session_state.df2_cluster.head(20))

        st.markdown("---")
        # ======== NAME CLUSTERS ========
        name_map = {
            0: "Cụm 0: Xe cao cấp",
            1: "Cụm 1: Xe phổ thông",
            2: "Cụm 2: Xe tầm trung",
            3: "Cụm 3: Xe giá rẻ"
        }

        st.write("Tên cụm gợi ý:", name_map)
        # ---------------- DỰ ĐOÁN CỤM CHO XE MỚI ----------------
        st.subheader("🔮 Dự đoán cụm cho xe mới")

        if st.session_state.cluster_model is None:
            st.warning("⚠ Bạn cần chạy clustering trước!")
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

                # Encode brand
                new_brand = encoder.transform(pd.DataFrame({"Thương hiệu": [brand]}))

                # Scale numeric bằng PowerTransformer
                new_numeric = scaler.transform([[gia, km, cc, year]])

                X_new = np.concatenate([new_numeric, new_brand.values], axis=1)

                # Predict
                if hasattr(model, "predict"):
                    cluster_id = model.predict(X_new)[0]
                else:
                    # Agglomerative fallback
                    centroids = np.vstack([
                        st.session_state.X2_scaled[st.session_state.labels == c].mean(axis=0)
                        for c in range(4)
                    ])
                    cluster_id = np.argmin(np.linalg.norm(centroids - X_new, axis=1))

                cluster_label = cluster_names.get(cluster_id, "Không rõ")

                st.success(f"✔ Xe của bạn thuộc **Cụm {cluster_id} – {cluster_label}!**")

# =============== PAGE: VISUALIZATION =============== #
elif page == "Visualization":
    st.subheader("📊 Visualization Dashboard")

    # Histogram
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(df["Giá"].dropna(), kde=True, ax=ax1)
    st.pyplot(fig1)

    # WordCloud
    text = " ".join(df["Tiêu đề"].astype(str))
    wc = WordCloud(width=900, height=400, background_color="white").generate(text)
    fig2, ax2 = plt.subplots(figsize=(9,4))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

# =============== PAGE: TASK ASSIGNMENT =============== #
elif page == "Task assignment":
    st.subheader("📋 Task Assignment")

    st.markdown("""
        ### 🧑‍💻 Bảng phân công công việc
        
        | Thành viên         | Công việc |
        |--------------------|-----------|
        | **Nguyễn Duy Thanh** | GUI for Recommendation System and Clustering |
        | **Nguyễn Thái Bình** | GUI for Price Prediction and Anomaly Detection |
    """)


# ===================== FOOTER ===================== #
st.sidebar.markdown("---")
# Load ảnh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== AVATAR ===================== #
avatar_path = os.path.join(BASE_DIR, "avatar.jpg")
avatar = Image.open(avatar_path)

# --- THÔNG SỐ ---
offset_ratio = 0.10   # dịch xuống 15% chiều cao ảnh (có thể chỉnh 0.10–0.25)

# --- Crop top nhưng dịch xuống ---
w, h = avatar.size
size = min(w, h)

# Tính offset theo tỉ lệ chiều cao
offset = int(size * offset_ratio)

left   = (w - size) / 2
top    = offset
right  = (w + size) / 2
bottom = offset + size

# Đảm bảo không vượt quá ảnh thật
bottom = min(bottom, h)

avatar = avatar.crop((left, top, right, bottom))

# --- Resize sắc nét ---
avatar = avatar.resize((80, 80), Image.LANCZOS)

# --- Hiển thị ---
st.sidebar.image(avatar, width=80, use_column_width=False)

# --- Footer ---
st.sidebar.write("Designed by **Duy-Thanh Nguyen**")
st.sidebar.write("Email: duythanh200620@gmail.com")
