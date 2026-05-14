from bson import ObjectId
import os
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv

model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

mongo_url = os.getenv("MONGO_URL")

try:
    client = MongoClient(mongo_url)
    db = client["ITJOBS"]
    jobs = db.jobs
    skills = db.skills
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise RuntimeError("Failed to connect to MongoDB. Check MONGO_URL and your connection.")

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
    Generate embeddings for ALL jobs, không phân biệt đã có hay chưa
    """
    # Lấy tất cả jobs
    query = {}
    
    total = jobs.count_documents(query)
    print(f"Total jobs in database: {total}")
    print(f"Generating embeddings for ALL {total} jobs...\n")

    # Lấy tất cả jobs, sắp xếp mới nhất trước
    all_jobs = jobs.find(query).sort("createdAt", -1)

    processed = 0
    batch = []

    for job in all_jobs:
        batch.append(job)
        if len(batch) >= batch_size:
            process_batch(batch)
            processed += len(batch)
            print(f"Processed {processed}/{total} jobs...")
            batch = []

    if batch:
        process_batch(batch)
        processed += len(batch)
        print(f"Processed {processed}/{total} jobs...")

    print(f"\n✔ Generated embeddings for {processed} jobs!")
    
    # Verify final count
    final_with_embedding = jobs.count_documents({"embedding": {"$exists": True}})
    final_without = jobs.count_documents({"embedding": {"$exists": False}})
    print(f"Final: {final_with_embedding} jobs with embedding, {final_without} without")


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
