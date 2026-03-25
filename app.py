from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_login import LoginManager, current_user
import os
from dotenv import load_dotenv

load_dotenv()
import uuid
import shutil
import requests
from werkzeug.utils import secure_filename
import numpy as np
import logging
import json
import time

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- OPTIONAL RAG MODULES ----------------
try:
    from pdf_to_vectors import pdf_to_vectors
    from txt_to_vectors import txt_to_vectors  
    logger.info("RAG modules loaded.")
except Exception:
    logger.exception("RAG modules not loaded.")

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-for-dev")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Database Setup
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("MYSQL_URI", "sqlite:///fallback.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from models import db, User
db.init_app(app)

# LoginManager Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'error'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from auth import auth_bp, bcrypt
bcrypt.init_app(app)
app.register_blueprint(auth_bp)

from dashboard import dashboard_bp
app.register_blueprint(dashboard_bp)

with app.app_context():
    db.create_all()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# ---------------- HELPERS ----------------
def classify(score):
    if score >= 0.75:
        return "Good Fit"
    elif score >= 0.5:
        return "Moderate Fit"
    else:
        return "Poor Fit"

def safe_json_load(raw):
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}")
        return {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- LOCAL LLM ----------------
def call_local_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=60
        )
        return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Local LLM failed: {e}")
        return None

# ---------------- SKILLS ----------------
SKILLS = ["python","java","javascript","react","node","docker","kubernetes","aws","sql","excel"]

def extract_skills(text):
    text = text.lower()
    return [s for s in SKILLS if s in text]

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.home'))
    return render_template('index.html')

from flask_login import login_required

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    files = request.files.getlist('files')
    jd = request.form.get('role_requirements')
    job_id = request.form.get('job_id')
    
    if not job_id:
        return redirect(url_for('dashboard.home'))

    uploaded = []
    for f in files:
        if allowed_file(f.filename):
            name = secure_filename(f.filename)
            unique = f"{uuid.uuid4()}_{name}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)
            uploaded.append({'unique': unique, 'original': f.filename, 'path': path})

    if not uploaded:
         from flask import flash
         flash('No valid files uploaded.', 'error')
         return redirect(url_for('dashboard.job_details', job_id=job_id))

    file_names = [f['unique'] for f in uploaded]
    process(file_names)
    results = match(jd)

    # Save candidates to database
    from models import db, Candidate
    from flask import flash, session
    
    recent_candidate_ids = []
    
    for r in results:
        original_name = next((u['original'] for u in uploaded if u['unique'] == r['resume']), r['resume'])
        path_ = next((u['path'] for u in uploaded if u['unique'] == r['resume']), "")
        
        ai_data = {
            "candidate_name": r.get("ai_candidate_name", ""),
            "fitness_score": r.get("ai_match_score", 0),
            "classification": r.get("ai_classification", ""),
            "experience_level": r.get("ai_experience_level", ""),
            "key_strengths": r.get("ai_strong_skills", []),
            "missing_critical_skills": r.get("ai_missing_skills", []),
            "project_relevance": r.get("ai_project_relevance", ""),
            "improvement_suggestions": r.get("ai_improvement_suggestions", []),
            "score_reason": r.get("ai_score_reason", "Analysis failed/unavailable. Using embedded matching scores.")
        }
        # Use the LLM's generated fitness score for the final ranking if it successfully analyzed the candidate, else fallback to RAG final score.
        ai_match_score_pct = float(r.get("ai_match_score", 0))
        final_score_calculated = ai_match_score_pct / 100.0 if ai_match_score_pct > 0 else r.get("final_score", 0.0)
        
        classification = r.get("ai_classification")
        if not classification:
            classification = r.get("classification", "Unknown")

        candidate = Candidate(
            job_id=job_id,
            filename=original_name,
            file_path=path_,
            match_score=final_score_calculated,
            classification=classification,
            ai_analysis=ai_data
        )
        db.session.add(candidate)
        db.session.flush() # get ID before commit
        recent_candidate_ids.append(candidate.id)
        
    db.session.commit()
    session['recent_candidates'] = recent_candidate_ids
    flash('Resumes processed and AI matches synchronized successfully!', 'success')
    return redirect(url_for('dashboard.job_details', job_id=job_id))

# ---------------- PROCESS ----------------
def process(files):
    import faiss, pickle

    shutil.rmtree('processed', ignore_errors=True)
    os.makedirs('processed', exist_ok=True)

    emb_all, chunks_all, meta_all = [], [], []

    for f in files:
        path = os.path.join('uploads', f)

        if f.endswith('.pdf'):
            emb, chunks, meta, _ = pdf_to_vectors(path)
        else:
            emb, chunks, meta = txt_to_vectors(path)

        emb_all.extend(emb)
        chunks_all.extend(chunks)
        meta_all.extend(meta)

    emb_np = np.array(emb_all).astype('float32')
    faiss.normalize_L2(emb_np)
    
    dim = emb_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_np)

    faiss.write_index(index, "processed/vectors.index")

    with open("processed/chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks_all, "meta": meta_all}, f)

# ---------------- MATCH ----------------
def match(jd):
    import faiss, pickle
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    index = faiss.read_index("processed/vectors.index")
    data = pickle.load(open("processed/chunks.pkl", "rb"))

    chunks = data["chunks"]
    meta = data["meta"]

    available_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
    chosen_model = "models/text-embedding-004" if "models/text-embedding-004" in available_models else (available_models[-1] if available_models else "models/gemini-embedding-2-preview")
    
    result = genai.embed_content(
        model=chosen_model,
        content=[jd],
        task_type="retrieval_query"
    )
    query = np.array(result['embedding']).astype("float32")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    faiss.normalize_L2(query)

    scores, idxs = index.search(query, min(50, len(chunks)))

    resume_scores = {}
    resume_chunks = {}

    skills_jd = extract_skills(jd)

    for s, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        name = meta[i]["source_file"]

        resume_scores.setdefault(name, []).append(s)
        resume_chunks.setdefault(name, []).append(chunks[i])

    ranked = []

    for name, score_list in resume_scores.items():
        top_scores = sorted(score_list, reverse=True)[:5]
        avg = float(np.mean(top_scores))

        text = " ".join(resume_chunks[name])
        skills_res = extract_skills(text)

        matched = [s for s in skills_jd if s in skills_res]
        missing = [s for s in skills_jd if s not in skills_res]

        skill_score = len(matched) / len(skills_jd) if skills_jd else 1
        final = (avg * 0.6) + (skill_score * 0.4)

        ranked.append({
            "resume": name,
            "final_score": round(final, 3),
            "classification": classify(final),
            "skills_matched": matched,
            "skills_missing": missing,
            "confidence": round(skill_score * 100, 2),
            "top_chunks": resume_chunks[name][:3],

            "ai_candidate_name": "",
            "ai_match_score": 0,
            "ai_classification": "",
            "ai_experience_level": "",
            "ai_strong_skills": [],
            "ai_missing_skills": [],
            "ai_project_relevance": "",
            "ai_improvement_suggestions": [],
            "ai_score_reason": "Analyzing..."
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # ---------------- AI ANALYSIS ----------------
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def analyze_candidate(r):
        resume_text = "\n".join(r["top_chunks"])[:3000]

        prompt = f"""
Return ONLY valid JSON.

Job:
{jd}

Resume:
{resume_text}

{{
 "candidate_name": "First Last",
 "fitness_score": 0-100,
 "classification": "",
 "experience_level": "",
 "key_strengths": [],
 "missing_critical_skills": [],
 "project_relevance": "",
 "improvement_suggestions": [],
 "score_reason": ""
}}
"""

        # Disable local LLM check to prevent 20-second timeout delays when Ollama isn't running.
        # raw = call_local_llm(prompt)
        raw = ""

        if not raw:
            try:
                # Try gemini-2.5-pro first (highest quality)
                model = genai.GenerativeModel("gemini-2.5-pro")
                res = model.generate_content(prompt)
                raw = res.text.strip()
                logger.info("Gemini 2.5 Pro used successfully")
            except Exception as e:
                logger.error(f"Gemini 2.5 Pro failed: {e}")
                try:
                    # Fallback to gemini-2.5-flash
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    res = model.generate_content(prompt)
                    raw = res.text.strip()
                    logger.info("Gemini 2.5 Flash fallback used")
                except Exception as e2:
                    logger.error(f"Gemini 2.5 Flash failed: {e2}")
                    raw = ""

        # robust json extraction
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if match:
            raw = match.group(1)
        elif raw.startswith("{") and raw.endswith("}"):
            pass # already JSON

        data = safe_json_load(raw)

        r["ai_candidate_name"] = data.get("candidate_name", "")
        r["ai_match_score"] = float(data.get("fitness_score", 0))
        r["ai_classification"] = data.get("classification", "")
        r["ai_experience_level"] = data.get("experience_level", "")
        r["ai_strong_skills"] = data.get("key_strengths", [])
        r["ai_missing_skills"] = data.get("missing_critical_skills", [])
        r["ai_project_relevance"] = data.get("project_relevance", "")
        r["ai_improvement_suggestions"] = data.get("improvement_suggestions", [])
        r["ai_score_reason"] = data.get("score_reason", "")

    # Execute all 5 AI analyses in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_candidate, r) for r in ranked[:5]]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Thread failed: {e}")

    return ranked

# ---------------- RUN ----------------
if __name__ == '__main__':
    logger.info("Server started.")
    app.run(debug=True, use_reloader=False, port=500)