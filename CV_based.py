from fastapi import FastAPI, UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import pdfplumber
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
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


def convert_objectid_to_str(obj):
    """Recursively convert ObjectId to string for JSON serialization"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    return obj


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


def generate_job_embedding(text):
    """Generate embedding from text when job doesn't have pre-computed embedding"""
    try:
        emb_full = model.encode(text)
        return emb_full.tolist()
    except:
        return np.zeros(384).tolist()


def prepare_job_skills(job):
    """Extract job skills with fallback for old jobs without structured fields"""
    # Prioritize structured fields
    must_have = job.get("mustHaveSkills", [])
    optional = job.get("optionalSkills", [])
    domain = job.get("domainKnowledge", [])
    
    # Fallback: extract from requirements text if structured fields are empty
    if not must_have and not optional and not domain:
        requirements_text = " ".join(job.get("requirements", []))
        if not requirements_text:
            requirements_text = job.get("description", "")
        
        if requirements_text:
            extracted = extract_skills(requirements_text)
            # Treat all extracted skills as must-have for old jobs
            must_have = extracted
    
    return {
        "must_have": set([s.lower() for s in must_have]) if must_have else set(),
        "optional": set([s.lower() for s in optional]) if optional else set(),
        "domain": set([s.lower() for s in domain]) if domain else set()
    }


@app.post("/recommend")
async def recommend_jobs(file: UploadFile = File(...)):
    try:
        # Extract text
        pdf_bytes = await file.read()
        text = extract_pdf_text(BytesIO(pdf_bytes))
        if not text or text.strip() == "":
            return {
                "error": "Không thể đọc nội dung PDF. Vui lòng kiểm tra file CV.",
                "skills_found": [],
                "recommendations": []
            }
        
        cv_skills = extract_skills(text)
        cv_skills_set = set(cv_skills)

        # Embedding cho CV
        if len(cv_skills) == 0:
            emb_skill = np.zeros(384)  # all-MiniLM-L6-v2 has 384 dimensions
        else:
            emb_skill = model.encode(" ".join(cv_skills))
        
        emb_full = model.encode(text)
        cv_emb = 0.7 * emb_skill + 0.3 * emb_full

        # Lấy tất cả embedding job kèm requirements/skills
        jobs = list(job_collection.find({}, {
            "title": 1, 
            "embedding": 1,
            "mustHaveSkills": 1,
            "optionalSkills": 1,
            "domainKnowledge": 1,
            "location": 1,
            "salary_raw": 1,
            "company": 1,
            "requirements": 1,
            "description": 1
        }))

        if len(jobs) == 0:
            return {
                "skills_found": cv_skills,
                "recommendations": [],
                "message": "Không có công việc nào trong hệ thống"
            }

        # Build job vectors with fallback for missing embeddings
        job_vectors = []
        job_embeddings_fallback = []  # Track which jobs used fallback embeddings
        
        for i, job in enumerate(jobs):
            embedding = job.get("embedding")
            if embedding:
                job_vectors.append(embedding)
                job_embeddings_fallback.append(False)
            else:
                # Fallback: generate embedding from text
                requirements_text = " ".join(job.get("requirements", []))
                if not requirements_text:
                    requirements_text = job.get("description", "")
                
                if requirements_text:
                    generated_emb = generate_job_embedding(requirements_text)
                    job_vectors.append(generated_emb)
                else:
                    job_vectors.append(np.zeros(384).tolist())
                job_embeddings_fallback.append(True)
        
        job_vectors = np.array(job_vectors)

        cv_emb = normalize([cv_emb])[0]
        job_vectors = normalize(job_vectors)
        
        # Tính similarity
        scores = cosine_similarity([cv_emb], job_vectors)[0]

        # Format kết quả với chi tiết matched skills
        results = []
        for job, score, used_fallback_emb in zip(jobs, scores, job_embeddings_fallback):
            # Prepare job skills with fallback logic
            job_skills = prepare_job_skills(job)
            job_required_skills = job_skills["must_have"]
            job_optional_skills = job_skills["optional"]
            job_domain = job_skills["domain"]
            job_all_skills = job_required_skills | job_optional_skills | job_domain
            
            matched_required = cv_skills_set & job_required_skills
            matched_optional = cv_skills_set & job_optional_skills
            matched_domain = cv_skills_set & job_domain
            matched_total = cv_skills_set & job_all_skills
            
            # Tính tỷ lệ match
            match_percentage = 0
            if len(job_all_skills) > 0:
                match_percentage = round((len(matched_total) / len(job_all_skills)) * 100, 1)
            
            # Tạo lý do giải thích
            reason_parts = []
            if len(matched_required) > 0:
                reason_parts.append(f"Match {len(matched_required)}/{len(job_required_skills)} kỹ năng bắt buộc")
            if len(matched_optional) > 0:
                reason_parts.append(f"Match {len(matched_optional)} kỹ năng tùy chọn")
            
            reason = "; ".join(reason_parts) if reason_parts else "Nội dung CV phù hợp với yêu cầu công việc"

            results.append({
                "id": str(job["_id"]),
                "title": str(job.get("title", "")),
                "location": convert_objectid_to_str(job.get("location")),
                "salary": str(job.get("salary_raw", "")),
                "company": str(job.get("company", "")),
                "score": float(score),
                "similarity_percentage": round(float(score) * 100, 1),
                "matched_skills": {
                    "required": list(matched_required),
                    "optional": list(matched_optional),
                    "domain": list(matched_domain),
                    "total_matched": len(matched_total),
                    "total_job_skills": len(job_all_skills)
                },
                "match_percentage": match_percentage,
                "reason": reason
            })

        return convert_objectid_to_str({
            "skills_found": cv_skills,
            "recommendations": sorted(results, key=lambda x: x["score"], reverse=True)[:100],
            "summary": f"Tìm được {len(cv_skills)} kỹ năng trong CV của bạn"
        })
    
    except Exception as e:
        print(f"Recommend error: {str(e)}")
        import traceback
        traceback.print_exc()
        return convert_objectid_to_str({
            "error": f"Lỗi xử lý: {str(e)}",
            "skills_found": [],
            "recommendations": []
        })
