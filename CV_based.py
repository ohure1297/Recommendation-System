from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from cv_utils import (
    convert_objectid_to_str,
    extract_text_from_file,
    extract_skills,
    extract_location_text,
    extract_city,
    parse_experience_text,
    parse_job_experience,
    prepare_job_skills,
    generate_job_embedding,
    generate_cv_embedding,
    check_location_match,
    model,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(os.getenv("MONGO_URL"))
db = client["ITJOBS"]
job_collection = db["jobs"]

@app.post("/recommend")
async def recommend_jobs(file: UploadFile = File(...)):
    try:
        if not any(file.filename.lower().endswith(ext) for ext in [".pdf", ".docx"]):
            return {"error": "File không hợp lệ. Vui lòng tải lên file CV định dạng PDF hoặc DOCX.", "skills_found": [], "recommendations": []}

        file_bytes = await file.read()
        text = extract_text_from_file(file_bytes=file_bytes, filename=file.filename)
        if not text or not text.strip():
            return {"error": "Không thể đọc nội dung file. Vui lòng kiểm tra file CV.", "skills_found": [], "recommendations": []}

        cv_skills = extract_skills(text)
        cv_skills_set = set(cv_skills)

        emb_skill = np.zeros(384) if len(cv_skills) == 0 else model.encode(" ".join(cv_skills))
        emb_full = model.encode(text)
        cv_emb = 0.7 * emb_skill + 0.3 * emb_full

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
            "description": 1,
            "experience": 1,
            "experienceLevel": 1,
            "work_location_detail": 1
        }))

        if not jobs:
            return {"skills_found": cv_skills, "recommendations": [], "message": "Không có công việc nào trong hệ thống"}

        job_vectors = []
        fallback_flags = []
        for job in jobs:
            emb = job.get("embedding")
            if emb and len(emb) == 384:
                job_vectors.append(emb)
                fallback_flags.append(False)
            else:
                reqs = " ".join(job.get("requirements", [])) or job.get("description", "")
                job_vectors.append(generate_job_embedding(reqs) if reqs else np.zeros(384).tolist())
                fallback_flags.append(True)

        job_vectors = np.array(job_vectors)
        cv_emb = normalize([cv_emb])[0]
        job_vectors = normalize(job_vectors)
        scores = cosine_similarity([cv_emb], job_vectors)[0]

        cv_full_location = extract_location_text(text)
        cv_city = extract_city(cv_full_location) if cv_full_location else ""
        cv_experience_years = parse_experience_text(text)

        results = []
        for job, score, used_fallback in zip(jobs, scores, fallback_flags):
            skills = prepare_job_skills(job)
            job_all_skills = skills["must_have"] | skills["optional"] | skills["domain"]
            matched_total = cv_skills_set & job_all_skills
            skill_score = (len(matched_total) / len(job_all_skills) * 100) if job_all_skills else 0

            job_experience_years = parse_job_experience(job.get("experience") or job.get("experienceLevel"))
            experience_match = None
            experience_score = 50
            if cv_experience_years is not None and job_experience_years is not None:
                experience_match = cv_experience_years >= job_experience_years
                experience_score = 100 if experience_match else 0

            parts = []
            location_field = job.get("location")
            if isinstance(location_field, dict):
                parts.append(location_field.get("name", ""))
            elif isinstance(location_field, str):
                parts.append(location_field)
            parts.append(job.get("work_location_detail", ""))
            job_full_location = " | ".join([p for p in parts if p])
            job_city = extract_city(job_full_location) if job_full_location else ""
            location_match = check_location_match(cv_city, job_city) if cv_city and job_city else False
            location_score = 100 if location_match else (50 if (cv_city and job_city) else 0)

            match_percentage = round(skill_score * 0.7 + experience_score * 0.15 + location_score * 0.15, 1)

            reason_parts = []
            if len(cv_skills_set & skills["must_have"]) > 0:
                reason_parts.append(f"Match {len(cv_skills_set & skills['must_have'])}/{len(skills['must_have'])} kỹ năng bắt buộc")
            if experience_match is True:
                reason_parts.append("Kinh nghiệm đủ yêu cầu")
            elif experience_match is False:
                reason_parts.append("Kinh nghiệm có thể thấp hơn yêu cầu")
            if location_match:
                reason_parts.append("Địa điểm phù hợp")

            reason = "; ".join(reason_parts) if reason_parts else "Nội dung CV phù hợp với yêu cầu công việc"

            combined_score = float(score) + match_percentage / 100.0
            if experience_match is True:
                combined_score += 0.10
            if location_match:
                combined_score += 0.05

            results.append({
                "id": str(job["_id"]),
                "title": str(job.get("title", "")),
                "location": convert_objectid_to_str(job.get("location")),
                "salary": str(job.get("salary_raw", "")),
                "company": str(job.get("company", "")),
                "score": float(score),
                "similarity_percentage": round(float(score) * 100, 1),
                "match_percentage": match_percentage,
                "experience_required": job_experience_years,
                "experience_match": experience_match,
                "cv_experience_years": cv_experience_years,
                "location_match": location_match,
                "cv_location": cv_full_location,
                "cv_city": cv_city,
                "job_location_text": job_full_location,
                "job_city": job_city,
                "combined_score": round(combined_score, 4),
                "matched_skills": {
                    "required": list(cv_skills_set & skills["must_have"]),
                    "optional": list(cv_skills_set & skills["optional"]),
                    "domain": list(cv_skills_set & skills["domain"]),
                    "total_matched": len(matched_total),
                    "total_job_skills": len(job_all_skills),
                },
                "reason": reason,
            })

        return convert_objectid_to_str({
            "skills_found": cv_skills,
            "recommendations": sorted(results, key=lambda x: x["combined_score"], reverse=True)[:100],
            "summary": f"Tìm được {len(cv_skills)} kỹ năng trong CV của bạn",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return convert_objectid_to_str({"error": f"Lỗi xử lý: {str(e)}", "skills_found": [], "recommendations": []})

@app.post("/job-embedding")
async def create_job_embedding(payload: dict):
    try:
        text = payload.get("text", "")

        if not text.strip():
            return {"embedding": []}

        embedding = model.encode(text)

        return {
            "embedding": embedding.tolist()
        }

    except Exception as e:
        return {
            "error": str(e)
        }

@app.post("/cv-embedding")
async def create_cv_embedding(payload: dict):

    raw_text = payload.get("rawText", "")
    skills = payload.get("skills", [])

    if not raw_text:
        return {
            "success": False,
            "embedding": []
        }

    embedding = generate_cv_embedding(
        raw_text,
        skills
    )

    return {
        "success": True,
        "embedding": embedding
    }