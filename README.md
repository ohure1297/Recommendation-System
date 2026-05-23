# Recommendation System

Recommendation API for IT-JOBS-FINDER — dịch vụ gợi ý công việc dựa trên embedding của CV và mô tả công việc.

## Tổng quan
- Dùng `sentence-transformers` để tạo embedding cho CV và mô tả công việc.
- Lưu embedding trong MongoDB và tính cosine similarity để xếp hạng công việc.
- API được triển khai bằng `FastAPI` (module: `CV_based.py`).

## Yêu cầu
- Python 3.8+
- MongoDB (chuỗi kết nối lưu trong file `.env` dưới biến `MONGO_URL`)
- (Tùy chọn) GPU + CUDA để tăng tốc khi tạo embedding với `sentence-transformers`.

## Cài đặt (ví dụ Windows)
1. Tạo virtualenv và kích hoạt:

```
python -m venv venv
venv\Scripts\activate
```

2. Cài đặt dependencies:

```
pip install -r requirements.txt
```

3. Tạo file `.env` (hoặc cập nhật) với biến môi trường `MONGO_URL`.

> Lưu ý: `.env` có thể chứa thông tin nhạy cảm (username/password). Không commit thông tin nhạy cảm lên kho công khai.

## Docker

Nếu bạn muốn đóng gói dịch vụ recommend để chạy độc lập hoặc triển khai dễ dàng, có thể dùng Docker.

### Dockerfile
- Đã tạo `Dockerfile` trong thư mục `Recommendation-System`.
- Ứng dụng sẽ chạy bằng lệnh:

```
uvicorn CV_based:app --host 0.0.0.0 --port 8000
```

### docker-compose
- Đã thêm `docker-compose.yml` để chạy service recommend.
- Mặc định service expose cổng `8000`.

### Chạy bằng Docker Compose
1. Đảm bảo trong thư mục `Recommendation-System` có file `.env` với `MONGO_URL`.
2. Chạy:

```
docker compose up --build
```

3. Truy cập API:

```
http://localhost:8000/recommend
```

> Nếu bạn dùng MongoDB Atlas như hiện tại, không cần MongoDB local container. Chỉ cần `MONGO_URL` trỏ tới Atlas trong `.env`.

## Chạy các công cụ

- Tạo/generate embedding cho công việc (jobs):

```
python job_embedding_update.py
```

- Tạo/generate embedding cho CVs:

```
python cv_embedding_update.py
```

- Chạy API (FastAPI):

```
python -m uvicorn CV_based:app --reload --port 8000
```

## Endpoint chính
- `POST /recommend` — nhận file CV (`file` form field), hỗ trợ PDF và DOCX. Trả về JSON gồm `skills_found`, `recommendations` và thông tin gợi ý.

Ví dụ curl (tệp `cv.pdf`):

```
curl -F "file=@cv.pdf" http://localhost:8000/recommend
```

## Các file chính
- `CV_based.py` — FastAPI app (entrypoint module name: `CV_based:app`).
- `cv_utils.py` — hàm trích xuất text, skills, location, và helper cho embedding.
- `job_embedding_update.py` — script tạo embedding cho jobs và cập nhật vào MongoDB.
- `cv_embedding_update.py` — script tạo embedding cho CVs.
- `migrate_jobs.py` — script migrate cập nhật các job cũ (thêm embedding, trích xuất skills...).
- `cities.py` — danh sách và helper chuẩn hóa tên thành phố/tỉnh Việt Nam.

## Ghi chú triển khai
- `sentence-transformers` thường cần `torch` hoặc `tensorflow`. Tham khảo https://pytorch.org/ để cài `torch` phù hợp với hệ thống (CPU/CUDA).
- Nếu dùng UploadFile trong FastAPI, cần `python-multipart` đã được thêm trong `requirements.txt`.

## Liên hệ
- Đây là phần của dự án IT-JOBS-FINDER. Sửa đổi/ghi chú thêm nếu cần.