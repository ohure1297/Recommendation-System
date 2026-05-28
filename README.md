# 🤖 AI Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-7.x-47A248?style=for-the-badge\&logo=mongodb\&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?style=for-the-badge\&logo=redis\&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-5.4.0-37814A?style=for-the-badge\&logo=celery\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge\&logo=docker\&logoColor=white)

**Hệ thống Recommendation thông minh sử dụng Vector Embedding + Similarity Search + Async Background Processing**

[Xem Demo](#) · [Báo lỗi](#) · [Đóng góp](#đóng-góp)

</div>

---

# 📋 Mục lục

* [Tổng quan dự án](#-tổng-quan-dự-án)
* [Tính năng nổi bật](#-tính-năng-nổi-bật)
* [Tech Stack](#-tech-stack)
* [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
* [Sơ đồ tư duy](#-sơ-đồ-tư-duy)
* [Luồng xử lý chi tiết](#-luồng-xử-lý-chi-tiết)
* [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
* [Giải thích từng thành phần](#-giải-thích-từng-thành-phần)
* [Cài đặt & Chạy dự án](#-cài-đặt--chạy-dự-án)
* [API Documentation](#-api-documentation)
* [Design Patterns & Nguyên lý thiết kế](#-design-patterns--nguyên-lý-thiết-kế)
* [Những thách thức & Giải pháp](#-những-thách-thức--giải-pháp)

---

# 🎯 Tổng quan dự án

**AI Recommendation System** là hệ thống backend thông minh giúp phân tích dữ liệu người dùng, vector hóa nội dung, tính toán semantic similarity và đưa ra recommendation phù hợp bằng Machine Learning.

Hệ thống được thiết kế theo hướng production-ready backend architecture với khả năng xử lý bất đồng bộ, dễ mở rộng và tối ưu cho các bài toán:

* Job Recommendation
* Product Recommendation
* Content Recommendation
* Personalized Search

---

## 🧠 Vấn đề được giải quyết

| Vấn đề                                      | Giải pháp                              |
| ------------------------------------------- | -------------------------------------- |
| Recommendation không chính xác theo keyword | Semantic Embedding + Cosine Similarity |
| Dữ liệu lớn gây chậm hệ thống               | Async worker + Redis queue             |
| Dữ liệu thô khó xử lý                       | Cleaning + Normalization pipeline      |
| Khó scale recommendation engine             | Dockerized modular architecture        |
| Search không hiểu ngữ nghĩa                 | Sentence Transformer embeddings        |

---

# ✨ Tính năng nổi bật

* 🔍 **AI Recommendation Engine** — Gợi ý thông minh bằng vector similarity
* 🧩 **Vector Embedding Search** — Tìm kiếm semantic similarity với embedding vectors
* ⚡ **Async Processing** — Celery + Redis xử lý dữ liệu nền
* 📄 **Data Processing Pipeline** — Cleaning, normalization, feature extraction
* 🛡️ **Validation Layer** — Validate và sanitize dữ liệu đầu vào
* 🐳 **Docker-ready** — Toàn bộ infrastructure được containerized
* 📊 **Structured Logging** — Logging có cấu trúc để debug & monitor
* 🔄 **Scalable Architecture** — Thiết kế module hóa, dễ mở rộng AI pipeline

---

# 🛠 Tech Stack

## Core Framework

| Công nghệ       | Phiên bản | Vai trò            |
| --------------- | --------- | ------------------ |
| **FastAPI**     | 0.115.0   | REST API framework |
| **Uvicorn**     | 0.30.6    | ASGI server        |
| **Pydantic v2** | 2.8.2     | Data validation    |

---

## AI / ML

| Công nghệ                 | Phiên bản | Vai trò                |
| ------------------------- | --------- | ---------------------- |
| **sentence-transformers** | 3.1.1     | Generate embeddings    |
| **scikit-learn**          | Latest    | Similarity calculation |
| **NumPy**                 | Latest    | Numerical computation  |
| **Pandas**                | Latest    | Data preprocessing     |

---

## Infrastructure

| Công nghệ          | Phiên bản | Vai trò                    |
| ------------------ | --------- | -------------------------- |
| **Redis**          | 7.x       | Queue + caching            |
| **Celery**         | 5.4.0     | Background task processing |
| **MongoDB**        | 7.x       | Document database          |
| **Docker Compose** | —         | Container orchestration    |

---

# 🏗 Kiến trúc hệ thống

```text
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT / FRONTEND                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP Request
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                FASTAPI APPLICATION (Port 8000)              │
│                                                             │
│  ┌─────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  /health    │  │  /recommend    │  │  /upload-data  │   │
│  └──────┬──────┘  └────────┬───────┘  └────────┬───────┘   │
└─────────│──────────────────│───────────────────│───────────┘
          │                  │                   │
          ▼                  ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌───────────────────┐
│ Recommendation │  │ Similarity     │  │ Redis Queue       │
│ Engine         │  │ Search Service │  │ Background Tasks  │
└────────┬───────┘  └────────┬───────┘  └────────┬──────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     CELERY WORKER                           │
│                                                             │
│ 1. Data Extraction                                          │
│ 2. Cleaning & Normalize                                     │
│ 3. Chunking / Feature Processing                            │
│ 4. Embedding Generation                                     │
│ 5. Store Vector Embeddings                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 VECTOR DATABASE / STORAGE                   │
│                  MongoDB + Embedding Store                  │
└─────────────────────────────────────────────────────────────┘
```

---

# 🗺 Sơ đồ tư duy

## Tổng thể hệ thống

```text
                  AI RECOMMENDATION SYSTEM
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  📦 INFRASTRUCTURE     🔄 PROCESSING         🧠 AI CORE
        │                     │                     │
   ┌────┴────┐          ┌─────┴─────┐        ┌─────┴─────┐
   │ Docker  │          │ Celery    │        │ Embedding │
   │ Redis   │          │ Worker    │        │ Similarity│
   │ MongoDB │          │ Queue     │        │ Ranking   │
   └─────────┘          └─────┬─────┘        └─────┬─────┘
                               │                    │
                        ┌──────┴──────┐      ┌──────┴──────┐
                        │ Data Process │      │ Recommendation│
                        │ Pipeline     │      │ Engine        │
                        └──────────────┘      └──────────────┘
```

---

# 🔄 Luồng xử lý chi tiết

## 1. Data Processing Flow

```text
Raw Data Upload
      │
      ▼
Validation Layer
      │
      ▼
Text Cleaning
      │
      ▼
Feature Extraction
      │
      ▼
Embedding Generation
      │
      ▼
Store Vector Embeddings
```

---

## 2. Recommendation Flow

```text
User Query
      │
      ▼
Embedding Generation
      │
      ▼
Vector Similarity Search
      │
      ▼
Ranking Algorithm
      │
      ▼
Top-K Recommendation Results
```

---

# 📁 Cấu trúc thư mục

```text
recommendation-system/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── core/
│   ├── config.py
│   ├── dependencies.py
│   └── logger.py
│
├── routers/
│   ├── recommend.py
│   ├── upload.py
│   └── health.py
│
├── services/
│   │
│   ├── validation/
│   ├── processing/
│   ├── recommendation/
│   └── storage/
│
├── models/
│   └── schemas.py
│
└── workers/
    ├── celery_app.py
    └── processing_worker.py
```

---

# 🔍 Giải thích từng thành phần

## `embedding_service.py` — Embedding Engine

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(text)
```

Embedding giúp hệ thống recommendation hiểu ngữ nghĩa thay vì chỉ keyword matching.

---

## `similarity_service.py` — Recommendation Core

```python
similarity = cosine_similarity(user_vector, item_vectors)
```

Tính toán semantic similarity giữa user vector và item vectors.

---

## `processing_worker.py` — Async Background Processing

```python
@celery_app.task
def process_data(data):
    cleaned = cleaning_service.clean(data)
    embeddings = embedding_service.embed(cleaned)
    vector_service.store(embeddings)
```

Celery worker xử lý nền giúp API không bị block khi xử lý dữ liệu lớn.

---

# 🚀 Cài đặt & Chạy dự án

## Yêu cầu

* Docker & Docker Compose
* Python 3.11+
* MongoDB
* Redis

---

## 1. Clone repository

```bash
git clone https://github.com/ohure1297/Recommendation-System.git

cd Recommendation-System
```

---

## 2. Tạo file `.env`

```env
MONGODB_URL=mongodb://mongodb:27017

REDIS_URL=redis://redis:6379

EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 3. Chạy bằng Docker

```bash
docker-compose up --build
```

---

## 4. Chạy local

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload
```

---

# 📡 API Documentation

Sau khi chạy server:

```text
http://localhost:8000/docs
```

---

## Endpoints

### `GET /health`

```json
{
  "status": "ok"
}
```

---

### `POST /recommend`

Request:

```json
{
  "query": "Backend Python Developer"
}
```

Response:

```json
{
  "results": [
    {
      "id": "job_01",
      "score": 0.92
    }
  ]
}
```

---

# 🎨 Design Patterns & Nguyên lý thiết kế

## 1. Clean Architecture

```text
Routers → Services → Models
```

---

## 2. Separation of Concerns

* `embedding_service` → generate embeddings
* `similarity_service` → similarity calculation
* `ranking_service` → recommendation ranking

---

## 3. Dependency Injection

Inject dependencies qua FastAPI `Depends()` để dễ test và mở rộng.

---

## 4. Async-first

```python
async def recommend(query: str):
    embedding = await embedding_service.embed(query)
    results = await vector_service.search(embedding)

    return results
```

---

# 💡 Những thách thức & Giải pháp

| Thách thức                     | Giải pháp                |
| ------------------------------ | ------------------------ |
| Recommendation thiếu chính xác | Semantic embeddings      |
| Blocking API                   | Async worker             |
| Dữ liệu lớn xử lý chậm         | Redis queue + Celery     |
| Search không hiểu ngữ nghĩa    | Vector similarity search |
| Hệ thống khó scale             | Dockerized architecture  |

---

# 📈 Hướng phát triển

* [ ] Hybrid Recommendation System
* [ ] Deep Learning Recommendation Models
* [ ] Real-time Recommendation
* [ ] Recommendation Analytics Dashboard
* [ ] CI/CD Pipeline
* [ ] Kubernetes Deployment

---

# 👤 Tác giả

**Ohure1297**

[![GitHub](https://img.shields.io/badge/GitHub-ohure1297-181717?style=flat-square\&logo=github)](https://github.com/ohure1297)

---
