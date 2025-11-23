# 🏍️ **Motorcycle Recommendation & Clustering Dashboard ** 
Ứng dụng phân tích dữ liệu xe máy cũ kết hợp **Recommendation System**, **Clustering**, và **Visualization**.  
Hỗ trợ **hai chế độ sử dụng**: *Người dùng* và *Quản trị*, với cơ chế quản lý dữ liệu tập trung bằng RAM.

---

## 🚀 **1. Giới thiệu ứng dụng**

Ứng dụng được xây dựng nhằm giải quyết 3 vấn đề thường gặp trên các nền tảng rao vặt xe máy như Chợ Tốt:

1. Người mua khó tìm đúng xe mong muốn do số lượng tin đăng lớn  
2. Giá xe dao động mạnh giữa các bài đăng  
3. Người bán khó nhận biết nhóm khách hàng phù hợp  

Ứng dụng hỗ trợ 4 module chính:

### 🔍 1.1 Hệ thống gợi ý xe (Recommendation System)
- Sử dụng **TF-IDF + Cosine Similarity**
- Gợi ý xe tương tự theo:
  - Tiêu đề danh mục
  - Từ khóa do người dùng nhập
- Hiển thị mức độ tương đồng bằng Cosine Similarity

---

### 📦 1.2 Phân cụm xe máy (Clustering)
Thuật toán hỗ trợ:
- **KMeans**
- **Gaussian Mixture Model**
- **Agglomerative Clustering**

Dựa trên các thuộc tính:  
**Giá — Km đã đi — Dung tích — Năm đăng ký — Thương hiệu**

Có thể:
- Khảo sát số cụm (Elbow + Silhouette)
- Chạy phân cụm
- Đặt tên cụm
- Dự đoán cụm cho xe mới
- Xem bảng thống kê cụm đã cập nhật

---

### 🎨 1.3 Trực quan hóa
- WordCloud từ tiêu đề
- Histogram phân bố giá
- Countplot thương hiệu
- PCA 2D scatterplot

---

### 🔐 1.4 Chế độ Người dùng & Quản trị (Role-based UI)

| Chức năng | Người dùng | Quản trị |
|----------|------------|----------|
| Xem các trang mô tả | ✔️ | ✔️ |
| Upload dữ liệu | ❌ | ✔️ |
| Khảo sát số cụm | ❌ | ✔️ |
| Chạy phân cụm | ❌ | ✔️ |
| Đặt tên cụm | ❌ | ✔️ |
| Xem kết quả phân cụm | ✔️ | ✔️ |
| Gợi ý xe | ✔️ | ✔️ |
| Dự đoán cụm cho xe mới | ✔️ | ✔️ |

---

## ⚙️ **2. Cài đặt môi trường**

### 2.1 Clone repository
```bash
git clone https://github.com/Chanh206/Recomendation_system_Clustering_App.git
cd Recomendation_system_Clustering_App
```
### 2.2 Cài đặt thư viện cần thiết
Ứng dụng yêu cầu một số thư viện liên quan đến xử lý dữ liệu, NLP tiếng Việt, máy học và Streamlit.

Chạy lệnh sau để cài đặt toàn bộ thư viện:

```bash
pip install -r requirements.txt
```
## ▶️ **3. Chạy ứng dụng**
```bash
streamlit run main_code.py
```
Ứng dụng sẽ mở tại: http://localhost:8501/

## 📁 **4. Cấu trúc thư mục chính**
```arduino
📦 Recomendation_system_Clustering_App
 ┣ 📄 main_code.py
 ┣ 📄 requirements.txt
 ┣ 📄 Cosine_similarity_matrix.pkl
 ┣ 📄 vietnamese-stopwords.txt
 ┣ 📄 emojicon.txt
 ┣ 📄 wrong-word.txt
 ┣ 📄 avatar.jpg
 ┣ 📄 avatar_2.jpg
 ┣ 📄 Logo_ChoTot.png
 ┗ 📄 README.md
```

## 🔧 **5. Các tính năng nổi bật**
- Gợi ý xe theo từ khóa hoặc danh mục
- Kho dữ liệu được upload một lần duy nhất (dùng RAM, không ghi file)
- Phân quyền UI rõ ràng giữa User/Admin
- PCA visualization giúp xem cụm trực quan
- Tự động tính toán Silhouette Score
- Đặt tên cụm để dễ hiểu hơn
- Dự đoán cụm cho xe mới

## 👨‍💻 **6. Tác giả & Liên hệ**

Designed by:
- Nguyễn Duy Thanh
    + Email: duythanh200620@gmail.com
- Nguyễn Thái Bình
    + Email: thaibinh782k1@gmail.com