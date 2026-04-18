from bson import ObjectId
import os
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

model = SentenceTransformer("all-MiniLM-L6-v2")

client = MongoClient(os.getenv("MONGO_URL"))
db = client["ITJOBS"]
jobs = db["jobs"]
skills = db["skills"]


def get_skill_names(skill_ids):
    obj_ids = [ObjectId(s) for s in skill_ids if ObjectId.is_valid(s)]
    docs = skills.find({"_id": {"$in": obj_ids}})
    return [doc["name"] for doc in docs]


def merge_skills(job):
    """Tạo skills nếu job không có trường này."""
    if "skills" in job and isinstance(job["skills"], list) and len(job["skills"]) > 0:
        return job["skills"]

    merged = []

    for field in ["mustHaveSkills", "optionalSkills", "domainKnowledge", "languages"]:
        value = job.get(field, [])
        if isinstance(value, list):
            merged.extend(value)

    return merged


def update_all_job_embeddings(batch_size=50):
    """
    Chỉ update job:
    - Không có embedding
    - HOẶC không có skills
    """
    query = {
        "$or": [
            {"embedding": {"$exists": False}},
            {"skills": {"$exists": False}},
            {"skills": {"$size": 0}}
        ]
    }

    total = jobs.count_documents(query)
    print(f"Need to update: {total} jobs")

    all_jobs = jobs.find(query)

    processed = 0
    batch = []

    for job in all_jobs:
        batch.append(job)
        if len(batch) >= batch_size:
            process_batch(batch)
            processed += len(batch)
            print(f"Processed {processed}/{total}")
            batch = []

    if batch:
        process_batch(batch)
        processed += len(batch)
        print(f"Processed {processed}/{total}")

    print("Updated embeddings done ✔")


def process_batch(job_batch):
    for job in job_batch:
        merged_skills = merge_skills(job)

        # Xử lý skill name
        if merged_skills and all(ObjectId.is_valid(s) for s in merged_skills):
            skill_names = get_skill_names(merged_skills)
        else:
            skill_names = merged_skills

        skills_text = " ".join(skill_names)

        req = " ".join(job.get("requirements", [])) if isinstance(job.get("requirements"), list) else job.get("requirements", "")
        desc = job.get("description", "")

        emb_req = model.encode(req or "")
        emb_desc = model.encode(desc or "")
        emb_skill = model.encode(skills_text or "")

        job_emb = 0.5 * emb_req + 0.3 * emb_desc + 0.2 * emb_skill

        update_data = {
            "embedding": job_emb.tolist(),
            "skills": merged_skills
        }

        jobs.update_one(
            {"_id": job["_id"]},
            {"$set": update_data}
        )


if __name__ == "__main__":
    print("running main...")
    update_all_job_embeddings()
