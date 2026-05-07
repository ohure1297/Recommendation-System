"""
Migration script to update old jobs with missing fields:
- embedding (from requirements text)
- mustHaveSkills (extracted from requirements)
- optionalSkills (empty for old jobs)
- domainKnowledge (empty for old jobs)

Usage:
    python migrate_jobs.py
"""

import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import numpy as np
from tqdm import tqdm

# Initialize
model = SentenceTransformer("all-MiniLM-L6-v2")
client = MongoClient(os.getenv("MONGO_URL"))
db = client["ITJOBS"]
job_collection = db["jobs"]
skills_collection = db["skills"]


def load_skill_dict():
    """Load skill dictionary from database"""
    skills = skills_collection.find({}, {"name": 1})
    return [s["name"].lower() for s in skills]


skill_dict = load_skill_dict()
print(f"Loaded {len(skill_dict)} skills")


def extract_skills(text):
    """Extract skills from text"""
    text_lower = text.lower()
    skills = []
    for skill in skill_dict:
        if skill in text_lower or fuzz.partial_ratio(skill, text_lower) > 85:
            skills.append(skill)
    return list(set(skills))


def generate_embedding(text):
    """Generate embedding from text"""
    try:
        emb = model.encode(text)
        return emb.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(384).tolist()


def migrate_job(job):
    """Migrate single job document"""
    job_id = job["_id"]
    updated = False
    updates = {}
    
    # Check if embedding is missing
    if not job.get("embedding"):
        requirements_text = " ".join(job.get("requirements", []))
        if not requirements_text:
            requirements_text = job.get("description", "")
        
        if requirements_text:
            updates["embedding"] = generate_embedding(requirements_text)
            updated = True
            print(f"  - Generated embedding")
    
    # Check if mustHaveSkills is missing
    if not job.get("mustHaveSkills"):
        requirements_text = " ".join(job.get("requirements", []))
        if not requirements_text:
            requirements_text = job.get("description", "")
        
        if requirements_text:
            extracted_skills = extract_skills(requirements_text)
            updates["mustHaveSkills"] = extracted_skills
            updated = True
            print(f"  - Extracted {len(extracted_skills)} skills: {extracted_skills[:5]}...")
    
    # Set optionalSkills to empty array if missing
    if not job.get("optionalSkills"):
        updates["optionalSkills"] = []
        updated = True
    
    # Set domainKnowledge to empty array if missing
    if not job.get("domainKnowledge"):
        updates["domainKnowledge"] = []
        updated = True
    
    # Update document if any changes were made
    if updated:
        job_collection.update_one(
            {"_id": job_id},
            {"$set": updates}
        )
    
    return updated


def main():
    print("\n=== Starting job migration ===\n")
    
    # Count total jobs
    total_jobs = job_collection.count_documents({})
    print(f"Total jobs to process: {total_jobs}\n")
    
    if total_jobs == 0:
        print("No jobs to migrate")
        return
    
    # Find jobs with missing fields
    jobs_missing_embedding = job_collection.count_documents({"embedding": {"$exists": False}})
    jobs_missing_skills = job_collection.count_documents({"mustHaveSkills": {"$exists": False}})
    
    print(f"Jobs missing embedding: {jobs_missing_embedding}")
    print(f"Jobs missing mustHaveSkills: {jobs_missing_skills}\n")
    
    # Fetch all jobs
    jobs = list(job_collection.find({}))
    
    # Migrate each job
    migrated_count = 0
    for i, job in enumerate(tqdm(jobs, desc="Migrating jobs")):
        job_title = job.get("title", "Unknown")
        if migrate_job(job):
            migrated_count += 1
    
    print(f"\n=== Migration complete ===")
    print(f"Total jobs migrated: {migrated_count}/{total_jobs}")
    
    # Verify results
    jobs_with_embedding = job_collection.count_documents({"embedding": {"$exists": True}})
    jobs_with_skills = job_collection.count_documents({"mustHaveSkills": {"$exists": True}})
    
    print(f"\nAfter migration:")
    print(f"  - Jobs with embedding: {jobs_with_embedding}/{total_jobs}")
    print(f"  - Jobs with mustHaveSkills: {jobs_with_skills}/{total_jobs}")


if __name__ == "__main__":
    main()
