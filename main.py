from fastapi import FastAPI, UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import pdfplumber
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import numpy as np
from bson import ObjectId

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

client = MongoClient(os.getenv("MONGO_URL"))
db = client["ITJOBS"]
job_collection = db["jobs"]
skills_collection = db["skills"]


def load_skill_dict():
    skills = skills_collection.find({}, {"name": 1})
    return [s["name"].lower() for s in skills]

skill_dict = load_skill_dict()


def extract_pdf_text(file_like):
    text = ""
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_skills(text):
    text_lower = text.lower()
    skills = []
    for skill in skill_dict:
        if skill in text_lower or fuzz.partial_ratio(skill, text_lower) > 85:
            skills.append(skill)
    return list(set(skills))


@app.post("/recommend")
async def recommend_jobs(file: UploadFile = File(...)):
    # 1️⃣ Extract text
    pdf_bytes = await file.read()
    text = extract_pdf_text(BytesIO(pdf_bytes))
    cv_skills = extract_skills(text)

    # 2️⃣ Embedding cho CV
    emb_skill = model.encode(" ".join(cv_skills))
    emb_full = model.encode(text)
    cv_emb = 0.7 * emb_skill + 0.3 * emb_full

    # 3️⃣ Lấy tất cả embedding job (NHANH)
    jobs = list(job_collection.find({}, {"title": 1, "embedding": 1}))

    job_vectors = np.array([j["embedding"] for j in jobs])

    # 4️⃣ Tính similarity
    scores = cosine_similarity([cv_emb], job_vectors)[0]

    # 5️⃣ Format kết quả
    results = []
    for job, score in zip(jobs, scores):
        results.append({
            "id": str(job["_id"]),
            "title": job["title"],
            "location": job.get("location"),
            "salary": job.get("salary_raw") or job.get("salary_normalized"),
            "company": job.get("company"),
            "score": float(score)
        })


    return {
        "skills_found": cv_skills,
        "recommendations": sorted(results, key=lambda x: x["score"], reverse=True)
    }
