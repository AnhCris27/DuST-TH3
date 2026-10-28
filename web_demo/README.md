# 🚗 DUST Web Demo & Deployment Guide

> **DUST (Dual Swin Transformer for Multi-Modal Video and Time-Series Modeling)**  
> Published in IEEE/CVF CVPR 2024 Workshop (WAD).

Web demo tương tác thời gian thực cho hệ thống AI nhận diện và dự đoán mức độ rủi ro khi lái xe từ video 3D và cảm biến xe (Kinematics time-series).

---

## 🌟 Tính Năng Nổi Bật Của Bản Demo
- **4 Kịch bản thực tế có sẵn**: Lái xe an toàn (*Normal Driving*), Suýt va chạm (*Near Crash*), Cấn lề/va lốp (*Tire Strike*), Tai nạn (*Crash*).
- **Trực quan hóa đa phương thức đồng bộ**:
  - Khung hình camera hành trình 224×224 cùng thanh tua frame (Scrubber).
  - Đồ thị sóng 3 kênh cảm biến (Gia tốc ngang, Gia tốc dọc, Tốc độ xe) vẽ mượt mà bằng Canvas.
- **So sánh 3 nhánh mô hình**:
  - Nhánh 1: Kinematics Swin 1D (chỉ cảm biến)
  - Nhánh 2: Video Swin 3D (chỉ video)
  - Nhánh 3: DUST Multimodal Fusion (kết hợp cả hai)
- **Đánh giá rủi ro thời gian thực**: Chỉ số rủi ro (0-100%), khuyến nghị an toàn thông minh, đo độ trễ suy luận (ms).

---

## 1. Chạy Cục Bộ (Local Machine)

### Bước 1: Kích hoạt môi trường và cài thư viện
```bash
conda activate dust
pip install -r web_demo/requirements.txt
```

### Bước 2: Khởi động máy chủ
```bash
python web_demo/run_demo.py
```
Hoặc dùng Uvicorn trực tiếp:
```bash
cd web_demo
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Mở trình duyệt truy cập:
- **Giao diện Demo**: [http://localhost:8000](http://localhost:8000)
- **API Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 2. Triển Khai Với Docker

### Build Docker Image
```bash
docker build -t dust-ai-demo -f web_demo/Dockerfile .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name dust-demo-app dust-ai-demo
```
*(Nếu máy chủ có GPU NVIDIA, thêm cờ `--gpus all` để tăng tốc độ suy luận).*

---

## 3. Triển Khai Miễn Phí Lên Hugging Face Spaces

1. Tạo một Space mới trên [Hugging Face Spaces](https://huggingface.co/spaces).
2. Chọn **Space SDK: Docker** (Blank).
3. Đẩy mã nguồn repo lên Git của Space (đảm bảo file `Dockerfile` ở thư mục gốc hoặc chỉ định đường dẫn `web_demo/Dockerfile`).
4. Hugging Face sẽ tự động build container và cấp một URL public HTTPS hoàn toàn miễn phí.

---

## 4. Triển Khai Lên Render.com / Railway.app

1. Đẩy code lên GitHub repository của bạn.
2. Trên **Render** hoặc **Railway**, chọn *New Web Service* -> kết nối với GitHub repo.
3. Chọn môi trường **Docker** (Render/Railway sẽ tự động phát hiện `Dockerfile`).
4. Thiết lập Port: `8000`.
5. Bấm **Deploy** — hệ thống sẽ tự động build và cung cấp domain HTTPS công khai.
