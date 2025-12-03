# 🦅 Eagle Eye - CV Master

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-41cd52?style=for-the-badge&logo=qt&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5c3ee8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A comprehensive Computer Vision application for image processing, analysis, and face recognition.**

**Ứng dụng Thị giác Máy tính toàn diện cho xử lý ảnh, phân tích và nhận dạng khuôn mặt.**

_「那年冬天，以为同淋雪便可共白头」_

[Features | Tính năng](#-features--tính-năng) •
[Installation | Cài đặt](#-installation--cài-đặt) •
[Usage | Sử dụng](#-usage--sử-dụng) •
[Architecture | Kiến trúc](#-architecture--kiến-trúc) •
[Contributing | Đóng góp](#-contributing--đóng-góp)

</div>

---

## ✨ Features | Tính năng

### 🎨 Basic Operations | Thao tác cơ bản

- **Image I/O | Đọc/Ghi ảnh**: Load and save images in multiple formats (PNG, JPG, BMP, TIFF, PGM) | Đọc và lưu ảnh với nhiều định dạng
- **Color Conversions | Chuyển đổi màu**: RGB, Grayscale, HSV, LAB color spaces | Các không gian màu RGB, Xám, HSV, LAB
- **Histogram Analysis | Phân tích Histogram**: View and analyze image histograms | Xem và phân tích biểu đồ histogram
- **Histogram Equalization | Cân bằng Histogram**: Enhance image contrast | Tăng cường độ tương phản ảnh

### 🔍 Filters & Enhancement | Bộ lọc & Cải thiện

- **Smoothing Filters | Bộ lọc làm mịn**: Gaussian, Median, Bilateral, Box blur | Làm mờ Gaussian, Trung vị, Bilateral, Box
- **Sharpening | Làm sắc nét**: Laplacian, Unsharp masking | Laplacian, Mặt nạ unsharp
- **Edge Detection | Phát hiện cạnh**: Sobel, Canny, Prewitt operators | Toán tử Sobel, Canny, Prewitt
- **Custom Kernels | Kernel tùy chỉnh**: Apply user-defined convolution kernels | Áp dụng kernel tích chập tùy chỉnh
- **Live Preview | Xem trước trực tiếp**: Real-time filter preview with adjustable parameters | Xem trước bộ lọc theo thời gian thực

### 🔲 Morphological Operations | Phép toán hình thái học

- **Basic Operations | Phép toán cơ bản**: Erosion, Dilation, Opening, Closing | Co, Giãn, Mở, Đóng
- **Advanced | Nâng cao**: Gradient, Top-hat, Black-hat transforms | Gradient, Top-hat, Black-hat
- **Boundary Extraction | Trích xuất biên**: Extract object boundaries | Trích xuất đường biên đối tượng
- **Skeleton | Bộ xương**: Morphological skeletonization | Tạo bộ xương hình thái
- **Customizable | Tùy chỉnh**: Rectangle, Ellipse, Cross structuring elements | Phần tử cấu trúc: Chữ nhật, Elip, Chữ thập

### 📊 Frequency Domain | Miền tần số

- **FFT Analysis | Phân tích FFT**: 2D Fourier Transform visualization | Trực quan hóa biến đổi Fourier 2D
- **Frequency Filters | Bộ lọc tần số**: Low-pass, High-pass, Band-pass, Band-stop | Thông thấp, Thông cao, Thông dải, Chắn dải
- **Filter Types | Loại bộ lọc**: Ideal, Butterworth, Gaussian | Lý tưởng, Butterworth, Gaussian
- **Interactive | Tương tác**: Adjustable cutoff frequencies and filter orders | Điều chỉnh tần số cắt và bậc bộ lọc

### 🎯 Segmentation | Phân đoạn

- **Thresholding | Ngưỡng hóa**: Otsu's automatic, Manual, Adaptive | Tự động Otsu, Thủ công, Thích nghi
- **K-Means Clustering | Phân cụm K-Means**: Color-based image segmentation | Phân đoạn ảnh dựa trên màu sắc
- **Live Preview | Xem trước trực tiếp**: Real-time threshold adjustment | Điều chỉnh ngưỡng theo thời gian thực

### 👤 PCA & Face Recognition | PCA & Nhận dạng khuôn mặt

- **Eigenfaces**: Principal Component Analysis for face recognition | Phân tích thành phần chính cho nhận dạng khuôn mặt
- **Face Reconstruction | Tái tạo khuôn mặt**: Reconstruct faces with variable components | Tái tạo khuôn mặt với số thành phần thay đổi
- **Dataset Support | Hỗ trợ bộ dữ liệu**: Load face datasets from folder structure | Tải bộ dữ liệu khuôn mặt từ cấu trúc thư mục
- **Visualization | Trực quan hóa**: Mean face, Eigenfaces, Reconstruction comparison | Khuôn mặt trung bình, Eigenfaces, So sánh tái tạo

### 📦 JPEG Compression | Nén JPEG

- **DCT Visualization | Trực quan hóa DCT**: See Discrete Cosine Transform in action | Xem biến đổi Cosine rời rạc hoạt động
- **Quantization | Lượng tử hóa**: Understand how JPEG compression works | Hiểu cách nén JPEG hoạt động
- **Zig-zag Encoding | Mã hóa Zig-zag**: Visualize coefficient ordering | Trực quan hóa thứ tự hệ số
- **Quality Control | Điều khiển chất lượng**: Adjustable compression quality (1-100) | Điều chỉnh chất lượng nén (1-100)
- **Block Analysis | Phân tích khối**: Click any 8×8 block to analyze | Nhấp vào bất kỳ khối 8×8 nào để phân tích

### 📐 Geometric Transforms | Biến đổi hình học

- **Rotation | Xoay**: Rotate images with optional size preservation | Xoay ảnh với tùy chọn giữ kích thước
- **Scaling | Co giãn**: Scale X/Y independently or linked | Co giãn X/Y độc lập hoặc liên kết
- **Resize | Thay đổi kích thước**: Resize to specific dimensions | Thay đổi đến kích thước cụ thể
- **Flip | Lật**: Horizontal, Vertical, or Both | Ngang, Dọc, hoặc Cả hai

### 🖼️ Advanced Viewer | Trình xem nâng cao

- **Zoom & Pan | Thu phóng & Kéo**: Mouse wheel zoom, drag to pan | Cuộn chuột để zoom, kéo để di chuyển
- **Synchronized View | Xem đồng bộ**: Original and processed images sync together | Ảnh gốc và ảnh xử lý đồng bộ với nhau
- **Fit to View | Vừa khung nhìn**: Auto-fit images to window | Tự động căn ảnh vừa cửa sổ

---

## 🚀 Installation | Cài đặt

### Prerequisites | Yêu cầu

- Python 3.12 or higher | Python 3.12 trở lên
- [uv](https://github.com/astral-sh/uv) package manager (recommended) | Trình quản lý gói uv (khuyến nghị)

### Quick Start | Bắt đầu nhanh

```bash
# Clone the repository | Clone repo
git clone https://github.com/yourusername/computer-vision-app.git
cd computer-vision-app

# Install dependencies with uv | Cài đặt dependencies với uv
uv sync

# Run the application | Chạy ứng dụng
uv run python main.py
```

### Alternative: pip | Cách khác: pip

```bash
# Create virtual environment | Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies | Cài đặt dependencies
pip install -e .

# Run | Chạy
python main.py
```

### Dependencies | Thư viện phụ thuộc

| Package       | Version | Purpose                    | Mục đích                        |
| ------------- | ------- | -------------------------- | ------------------------------- |
| PySide6       | ≥6.6.0  | Qt6 GUI framework          | Framework giao diện Qt6         |
| OpenCV        | ≥4.8.0  | Computer vision operations | Các phép toán thị giác máy tính |
| NumPy         | ≥1.26.0 | Numerical computing        | Tính toán số học                |
| SciPy         | ≥1.11.0 | Scientific computing       | Tính toán khoa học              |
| Matplotlib    | ≥3.8.0  | Plotting and visualization | Vẽ biểu đồ và trực quan hóa     |
| scikit-learn  | ≥1.3.0  | Machine learning (PCA)     | Học máy (PCA)                   |
| QtAwesome     | ≥1.3.0  | Icon library               | Thư viện icon                   |
| pyqtdarktheme | ≥2.1.0  | Dark theme styling         | Giao diện tối                   |

---

## 📖 Usage | Sử dụng

### Loading Images | Tải ảnh

1. Click **"Load Image"** button or press `Ctrl+O` | Nhấn nút **"Load Image"** hoặc `Ctrl+O`
2. Select an image file (supports PNG, JPG, BMP, TIFF, PGM) | Chọn file ảnh (hỗ trợ PNG, JPG, BMP, TIFF, PGM)
3. Image appears in the left panel (Original) | Ảnh xuất hiện ở panel bên trái (Gốc)

### Applying Operations | Áp dụng phép toán

1. Select a tab from the sidebar (Basic, Filters, Morph, etc.) | Chọn tab từ sidebar (Basic, Filters, Morph, v.v.)
2. Adjust parameters using sliders and controls | Điều chỉnh tham số bằng slider và các điều khiển
3. Click **"Apply"** or enable **"Live Preview"** | Nhấn **"Apply"** hoặc bật **"Live Preview"**
4. Result appears in the right panel (Processed) | Kết quả xuất hiện ở panel bên phải (Đã xử lý)

### Saving Results | Lưu kết quả

1. Click **"Save Image"** button or press `Ctrl+S` | Nhấn nút **"Save Image"** hoặc `Ctrl+S`
2. Choose location and format | Chọn vị trí và định dạng
3. Processed image is saved | Ảnh đã xử lý được lưu

### Face Recognition (PCA Tab) | Nhận dạng khuôn mặt (Tab PCA)

1. Click **"Load Face Dataset"** | Nhấn **"Load Face Dataset"**
2. Select folder containing face images (e.g., `data/archive`) | Chọn thư mục chứa ảnh khuôn mặt (ví dụ: `data/archive`)
   ```
   archive/
   ├── s1/
   │   ├── 1.pgm
   │   └── ...
   ├── s2/
   └── ...
   ```
3. Wait for PCA computation | Đợi tính toán PCA
4. Use slider to adjust number of components | Dùng slider điều chỉnh số thành phần
5. Select different faces to see reconstruction | Chọn các khuôn mặt khác nhau để xem tái tạo

### JPEG Compression Analysis | Phân tích nén JPEG

1. Load any image | Tải bất kỳ ảnh nào
2. Go to **Compression** tab | Vào tab **Compression**
3. Click anywhere on the image to analyze that 8×8 block | Nhấp vào bất kỳ đâu trên ảnh để phân tích khối 8×8 đó
4. Adjust quality slider to see compression effects | Điều chỉnh slider chất lượng để xem hiệu ứng nén
5. View DCT coefficients, quantization, and zig-zag ordering | Xem hệ số DCT, lượng tử hóa và thứ tự zig-zag

---

## 🏗️ Architecture | Kiến trúc

```
computer-vision-app/
├── main.py                 # Application entry point | Điểm vào ứng dụng
├── pyproject.toml          # Project configuration | Cấu hình dự án
├── README.md               # This file | File này
│
├── core/                   # Core image processing modules | Module xử lý ảnh chính
│   ├── point.py           # Point operations | Phép toán điểm (brightness, contrast, gamma)
│   ├── filters.py         # Spatial filters | Bộ lọc không gian (blur, sharpen, edge)
│   ├── morphology.py      # Morphological operations | Phép toán hình thái học
│   ├── frequency.py       # Frequency domain processing | Xử lý miền tần số (FFT)
│   ├── segmentation.py    # Thresholding, K-means | Ngưỡng hóa, K-means
│   ├── pca.py             # PCA face recognition | Nhận dạng khuôn mặt PCA
│   ├── compression.py     # JPEG compression simulation | Mô phỏng nén JPEG
│   ├── geometry.py        # Geometric transformations | Biến đổi hình học
│   └── worker.py          # Background thread worker | Worker chạy nền
│
├── ui/                     # User interface modules | Module giao diện người dùng
│   ├── main_window.py     # Main application window | Cửa sổ chính
│   ├── control_panel.py   # Sidebar with tabs | Sidebar với các tab
│   ├── zoomable_viewer.py # Zoomable image viewer | Trình xem ảnh có zoom
│   ├── basic_tab.py       # Basic operations tab | Tab thao tác cơ bản
│   ├── filters_tab.py     # Filters tab | Tab bộ lọc
│   ├── morphology_tab.py  # Morphology tab | Tab hình thái học
│   ├── frequency_tab.py   # Frequency domain tab | Tab miền tần số
│   ├── segmentation_tab.py# Segmentation tab | Tab phân đoạn
│   ├── pca_tab.py         # PCA face recognition tab | Tab nhận dạng khuôn mặt PCA
│   ├── compression_tab.py # JPEG compression tab | Tab nén JPEG
│   └── geometry_tab.py    # Geometric transforms tab | Tab biến đổi hình học
│
├── data/                   # Sample datasets | Bộ dữ liệu mẫu
│   └── archive/           # ORL face database (40 subjects) | CSDL khuôn mặt ORL (40 người)
│
└── resources/              # Icons and assets | Icon và tài nguyên
```

### Design Patterns | Mẫu thiết kế

- **MVC Architecture | Kiến trúc MVC**: Separation of UI (views) and processing (models) | Tách biệt giao diện (views) và xử lý (models)
- **Signal-Slot Pattern | Mẫu Signal-Slot**: Qt's event handling for loose coupling | Xử lý sự kiện Qt cho liên kết lỏng
- **Worker Thread | Luồng Worker**: Heavy operations run in background threads | Các phép toán nặng chạy trong luồng nền
- **Tab-based UI | Giao diện dạng Tab**: Modular interface with switchable panels | Giao diện module với các panel chuyển đổi

---

## 🛠️ Development | Phát triển

### Running Tests | Chạy Tests

```bash
uv run pytest tests/ -v
```

### Code Style | Phong cách code

```bash
# Format code | Định dạng code
uv run black .

# Lint | Kiểm tra lỗi
uv run ruff check .
```

### Building Executable | Build file thực thi

```bash
# Using PyInstaller | Sử dụng PyInstaller
uv run pyinstaller --onefile --windowed main.py
```

---

## 🤝 Contributing | Đóng góp

Contributions are welcome! Please follow these steps:

Mọi đóng góp đều được chào đón! Vui lòng làm theo các bước sau:

1. Fork the repository | Fork repo
2. Create a feature branch | Tạo branch tính năng (`git checkout -b feature/amazing-feature`)
3. Commit your changes | Commit thay đổi (`git commit -m 'Add amazing feature'`)
4. Push to the branch | Push lên branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request | Mở Pull Request

### Guidelines | Hướng dẫn

- Follow PEP 8 style guide | Tuân theo hướng dẫn phong cách PEP 8
- Add docstrings to all functions | Thêm docstrings cho tất cả hàm
- Update README for new features | Cập nhật README cho tính năng mới
- Add tests for new functionality | Thêm tests cho chức năng mới

---

## 📄 License | Giấy phép

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Dự án này được cấp phép theo giấy phép MIT - xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 🙏 Acknowledgments | Lời cảm ơn

- [OpenCV](https://opencv.org/) - Computer vision library | Thư viện thị giác máy tính
- [Qt/PySide6](https://www.qt.io/) - GUI framework | Framework giao diện
- [ORL Face Database](https://cam-orl.co.uk/facedatabase.html) - Sample face dataset | Bộ dữ liệu khuôn mặt mẫu
- [QtAwesome](https://github.com/spyder-ide/qtawesome) - Icon library | Thư viện icon

---

<div align="center">

**Made with ❤️ and ☕**

_「那年冬天，以为同淋雪便可共白头」_

_Nếu cùng nhau đi dưới tuyết, liệu chúng ta có cùng đi đến bạc đầu..._

</div>
