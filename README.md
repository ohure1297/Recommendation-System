# 🤖 AI Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-7.x-47A248?style=for-the-badge\&logo=mongodb\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge\&logo=docker\&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-Embedding-orange?style=for-the-badge)

**Hệ thống Recommendation thông minh sử dụng Semantic Embedding + Vector Similarity Search cho bài toán gợi ý việc làm IT**

[Demo](#) · [Report Bug](https://github.com/ohure1297/Recommendation-System/issues) · [Contributing](#đóng-góp)

</div>

---

# 📋 Mục lục

* [Tổng quan dự án](#-tổng-quan-dự-án)
* [Tính năng nổi bật](#-tính-năng-nổi-bật)
* [Tech Stack](#-tech-stack)
* [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
* [Luồng recommendation](#-luồng-recommendation)
* [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
* [Giải thích thành phần](#-giải-thích-thành-phần)
* [Cài đặt & Chạy dự án](#-cài-đặt--chạy-dự-án)
* [API Documentation](#-api-documentation)
* [Design Patterns](#-design-patterns)
* [Challenges & Solutions](#-challenges--solutions)

---

# 🎯 Tổng quan dự án

**AI Recommendation System** là hệ thống recommendation thông minh giúp phân tích CV, vector hóa dữ liệu và tính toán semantic similarity nhằm gợi ý việc làm IT phù hợp với kỹ năng, kinh nghiệm và địa điểm của người dùng.

Hệ thống sử dụng:

* Sentence Transformers Embedding
* Cosine Similarity
* Vector-based Recommendation
* CV Parsing Pipeline
* Semantic Matching

Mục tiêu của dự án là cải thiện độ chính xác recommendation so với phương pháp keyword matching truyền thống.

---

## 🧠 Vấn đề được giải quyết

| Vấn đề                           | Giải pháp                                    |
| -------------------------------- | -------------------------------------------- |
| Keyword matching thiếu chính xác | Semantic Embedding                           |
| CV có format không đồng nhất     | CV preprocessing pipeline                    |
| Search không hiểu ngữ nghĩa      | Vector Similarity Search                     |
| Recommendation chưa cá nhân hóa  | Matching theo skills + experience + location |
| Dữ liệu location không đồng nhất | City normalization logic                     |

---

# ✨ Tính năng nổi bật

* 📄 CV Parsing & Processing
* 🔍 Semantic Job Recommendation
* 🧩 Vector Embedding Similarity Search
* 📍 Location-aware Recommendation
* ⚡ Fast Similarity Calculation
* 🐳 Dockerized Deployment
* 📊 Structured Recommendation Pipeline
* 🔄 Modular AI Processing Scripts

---

# 🛠 Tech Stack

## Core Framework

| Công nghệ | Vai trò            |
| --------- | ------------------ |
| FastAPI   | REST API framework |
| Uvicorn   | ASGI server        |
| Pydantic  | Data validation    |

---

## AI / Machine Learning

| Công nghệ             | Vai trò                       |
| --------------------- | ----------------------------- |
| sentence-transformers | Generate semantic embeddings  |
| scikit-learn          | Cosine similarity calculation |
| NumPy                 | Numerical computation         |
| Pandas                | Data preprocessing            |

---

## Database & Infrastructure

| Công nghệ      | Vai trò                       |
| -------------- | ----------------------------- |
| MongoDB        | Store jobs & CV data          |
| Docker         | Containerization              |
| Docker Compose | Multi-container orchestration |

---

# 🏗 Kiến trúc hệ thống

```text
┌────────────────────────────────────────────────────┐
│                CLIENT / FRONTEND                  │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│               FASTAPI APPLICATION                 │
│                                                    │
│   ┌─────────────┐      ┌────────────────────┐      │
│   │  Upload CV  │      │ Recommendation API │      │
│   └──────┬──────┘      └──────────┬─────────┘      │
└──────────│────────────────────────│────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────┐      ┌─────────────────────────┐
│  CV Processing   │      │ Similarity Calculation │
│  Pipeline        │      │ Cosine Similarity      │
└────────┬─────────┘      └──────────┬──────────────┘
         │                           │
         ▼                           ▼
┌────────────────────────────────────────────────────┐
│         Sentence Transformer Embeddings            │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│                    MongoDB                        │
│        CV Data + Job Data + Embeddings           │
└────────────────────────────────────────────────────┘
```

---

# 🔄 Luồng Recommendation

## CV Recommendation Pipeline

```text
CV Upload
    │
    ▼
Text Extraction
    │
    ▼
Data Cleaning
    │
    ▼
Skill Extraction
    │
    ▼
Experience Parsing
    │
    ▼
Location Normalization
    │
    ▼
Embedding Generation
    │
    ▼
Similarity Matching
    │
    ▼
Top-K Job Recommendation
```

---

## Semantic Matching Workflow

```text
User CV
    │
    ▼
Sentence Transformer
    │
    ▼
Vector Embedding
    │
    ▼
Cosine Similarity
    │
    ▼
Ranking Algorithm
    │
    ▼
Recommended Jobs
```

---

# 📁 Cấu trúc thư mục

```text
Recommendation-System/
│
├── .dockerignore
├── .gitignore
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── CV_based.py
├── cv_utils.py
├── cv_embedding_update.py
├── job_embedding_update.py
├── migrate_jobs.py
└── cities.py
```

---

# 🔍 Giải thích thành phần

## `CV_based.py` — Main Recommendation API

File trung tâm của hệ thống recommendation.

Chức năng:

* Upload & xử lý CV
* Recommendation pipeline
* Similarity matching
* API endpoints
* Response generation

---

## `cv_utils.py` — CV Processing Utilities

Chứa các utility functions:

* Extract text từ PDF
* Skill extraction
* Experience parsing
* Text preprocessing
* Data normalization

---

## `cv_embedding_update.py` — CV Embedding Pipeline

Sinh vector embeddings cho CV.

```python
embedding = model.encode(cv_text)
```

Embedding giúp hệ thống hiểu semantic meaning thay vì keyword matching đơn thuần.

---

## `job_embedding_update.py` — Job Embedding Pipeline

Tạo embeddings cho job descriptions để phục vụ recommendation.

Workflow:

```text
Job Description
      │
      ▼
Text Cleaning
      │
      ▼
Embedding Generation
      │
      ▼
Store Embedding
```

---

## `migrate_jobs.py` — Data Migration Script

Script phục vụ:

* Import jobs
* Data preprocessing
* Data normalization
* Database migration

---

## `cities.py` — Location Matching Logic

Xử lý location normalization và city matching.

Ví dụ:

```text
TP.HCM
Ho Chi Minh
HCM

→ Ho Chi Minh City
```

Giúp recommendation theo location chính xác hơn.

---

# 🚀 Cài đặt & Chạy dự án

## Yêu cầu

* Python 3.11+
* MongoDB
* Docker & Docker Compose

---

## 1. Clone repository

```bash
git clone https://github.com/ohure1297/Recommendation-System.git

cd Recommendation-System
```

---

## 2. Tạo virtual environment

```bash
python -m venv venv
```

---

## 3. Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Chạy hệ thống

```bash
uvicorn CV_based:app --reload
```

---

## 6. Chạy bằng Docker

```bash
docker-compose up --build
```

---

# 📡 API Documentation

Sau khi chạy server:

```text
http://localhost:8000/docs
```

---

# 🔌 API Endpoints

## `GET /`

Health check endpoint.

Response:

```json
{
  "message": "Recommendation API Running"
}
```

---

## `POST /recommend`

Request:

```json
{
  "cv_text": "Python backend developer with 3 years experience"
}
```

Response:

```json
{
  "recommendations": [
    {
      "job_title": "Backend Developer",
      "similarity_score": 0.91
    }
  ]
}
```

---

# 🎨 Design Patterns

## 1. Modular Script-based Architecture

Hệ thống được tổ chức theo từng processing modules riêng biệt:

* CV processing
* Embedding generation
* Similarity calculation
* Recommendation ranking

Giúp dễ maintain và mở rộng pipeline.

---

## 2. Separation of Concerns

| Module                  | Responsibility         |
| ----------------------- | ---------------------- |
| cv_utils.py             | CV processing          |
| cv_embedding_update.py  | CV embeddings          |
| job_embedding_update.py | Job embeddings         |
| migrate_jobs.py         | Data migration         |
| cities.py               | Location normalization |

---

## 3. Vector-based Recommendation

Recommendation dựa trên:

```text
Semantic Similarity
```

thay vì keyword matching truyền thống.

---

# 💡 Challenges & Solutions

| Challenges                           | Solutions                       |
| ------------------------------------ | ------------------------------- |
| CV format không đồng nhất            | CV preprocessing pipeline       |
| Recommendation thiếu chính xác       | Sentence Transformer embeddings |
| Search không hiểu ngữ nghĩa          | Vector similarity search        |
| Matching theo location khó chính xác | City normalization              |
| Recommendation ranking chưa tối ưu   | Similarity scoring              |

---

# 📈 Future Improvements

* [ ] Hybrid Recommendation System
* [ ] Deep Learning Ranking Model
* [ ] Real-time Recommendation
* [ ] Recommendation Analytics Dashboard
* [ ] Kubernetes Deployment
* [ ] CI/CD Pipeline
* [ ] Multi-language Recommendation
* [ ] Distributed Vector Search

---

# 👤 Author

**Ohure1297**

GitHub:
https://github.com/ohure1297

