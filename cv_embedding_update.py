from bson import ObjectId
import os
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np

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

def load_skill_dict():
    """Load skills from database for matching"""
    skills = skills.find({}, {"name": 1})
    return [s["name"].lower() for s in skills]

skill_dict = load_skill_dict()

def extract_skills_from_text(text):
    """Extract skills from CV text using database skills"""
    text_lower = text.lower()
    found_skills = []

    for skill in skill_dict:
        if skill in text_lower:
            found_skills.append(skill)

    return found_skills

def generate_cv_embeddings(model, batch_size=20):
    """
    Generate embeddings for parsed CVs
    Uses same logic as job embeddings but optimized for CV content
    """
    print("🚀 Starting CV embedding generation...")
    # Find CVs that need embeddings (no embedding field or empty)
    query = {
        "$or": [
            {"embedding": {"$exists": False}},
            {"embedding": {"$size": 0}},
            {"embedding": None}
        ]
    }

    total_cvs = db.RESUMES.count_documents(query)
    print(f"CVs needing embeddings: {total_cvs}")

    if total_cvs == 0:
        print("All CVs already have embeddings!")
        return

    all_cvs = db.RESUMES.find(query)
    processed = 0
    successful = 0

    for cv in all_cvs:
        try:
            # Get parsed resume data
            parsed_resume = db.PARSED_RESUMES.find_one({"resumeId": cv["_id"]})

            if not parsed_resume:
                print(f"No parsed data for CV: {cv.get('fileName', cv['_id'])}")
                continue

            raw_text = parsed_resume.get("rawText", "")
            skills = parsed_resume.get("skills", [])

            if not raw_text:
                print(f"⚠️  Empty text for CV: {cv.get('fileName', cv['_id'])}")
                continue

            # Extract additional skills from text if parsed skills are limited
            if len(skills) < 3:
                additional_skills = extract_skills_from_text(raw_text)
                skills.extend(additional_skills)
                skills = list(set(skills))  # Remove duplicates

            # Generate embeddings
            skills_text = " ".join(skills) if skills else ""

            # CV embedding strategy: 70% skills + 30% full text (same as jobs)
            emb_skill = model.encode(skills_text) if skills_text else np.zeros(384)
            emb_full = model.encode(raw_text)

            cv_embedding = 0.7 * emb_skill + 0.3 * emb_full

            # Update CV document
            update_data = {
                "embedding": cv_embedding.tolist(),
                "skills": skills,  # Update skills if we found more
                "embeddingGeneratedAt": {"$date": "new Date()"}  # MongoDB date
            }

            db.RESUMES.update_one(
                {"_id": cv["_id"]},
                {"$set": update_data}
            )

            successful += 1
            processed += 1

            if processed % 10 == 0:
                print(f"✅ Processed {processed}/{total_cvs} CVs")

        except Exception as e:
            print(f"❌ Error processing CV {cv.get('_id')}: {str(e)}")
            processed += 1

    print(f"EMBEDDING GENERATION COMPLETE")
    print(f"Total CVs processed: {processed}")
    print(f"Successful embeddings: {successful}")
    print(f"Success rate: {successful/processed*100:.1f}%" if processed > 0 else "0%")

    # Verification
    embedded_count = db.RESUMES.count_documents({"embedding": {"$exists": True, "$ne": []}})
    total_count = db.RESUMES.count_documents({})
    print(f"   CVs with embeddings: {embedded_count}/{total_count}")

def verify_embeddings():
    """Verify embedding quality"""
    print("\n🔍 Verifying embedding quality...")

    # Sample some CVs with embeddings
    sample_cvs = list(db.RESUMES.find(
        {"embedding": {"$exists": True, "$ne": []}}
    ).limit(5))

    if not sample_cvs:
        print("No CVs with embeddings found!")
        return

    print("Sample CVs with embeddings:")
    for cv in sample_cvs:
        embedding = cv.get("embedding", [])
        skills = cv.get("skills", [])
        print(f"   {cv.get('fileName', 'Unknown')}: {len(embedding)} dims, {len(skills)} skills")

def main():
    """Main function"""
    print("CV Embedding Generation Tool")
    print("=" * 50)

    # Check MongoDB connection
    try:
        collections = db.list_collection_names()
        print(f"MongoDB connected: {len(collections)} collections")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return

    # Initialize embedding model
    print("Loading sentence-transformers model...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Generate embeddings
    generate_cv_embeddings(model)

    # Verify
    verify_embeddings()

    print("\nCV embedding generation completed!")
    print("\nNext steps:")
    print("1. Test CV-to-job recommendations")
    print("2. Implement job-to-CV recommendations")
    print("3. Run evaluation metrics")

if __name__ == "__main__":
    main()