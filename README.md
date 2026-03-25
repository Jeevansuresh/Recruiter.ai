# Recruiter AI

## The Problem

Large companies running mass hiring drives face a brutal bottleneck: HR teams spend enormous amounts of time manually reviewing hundreds — sometimes thousands — of resumes per role. Most of that time is wasted on candidates who aren't a fit, while genuinely strong candidates often get overlooked simply because no one had time to read their resume carefully.

**Recruiter AI** eliminates that bottleneck. It replaces manual resume screening with an AI pipeline that reads, understands, and ranks every candidate against the job requirements — so HR teams stop drowning in CVs and start focusing only on the best fits. It doesn't just filter; it finds the strongest candidates intelligently, so companies hire smarter and faster.

## Features

- **Job Postings** — Create and manage job listings with full descriptions
- **Candidate Analysis** — RAG-based resume scoring against job requirements
- **Interview Scheduling** — Track and manage candidate interviews
- **AI Outreach** — Generate personalised outreach emails with Gemini
- **Analytics Dashboard** — Hiring pipeline metrics at a glance

## Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **AI**: Google Gemini API, FAISS vector store
- **Frontend**: Jinja2 templates, Vanilla CSS

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/recruiter-ai.git
cd recruiter-ai
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Run the app

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## Project Structure

```
.
├── app.py              # App factory & entry point
├── auth.py             # Authentication blueprint
├── dashboard.py        # Dashboard blueprint & routes
├── models.py           # SQLAlchemy models
├── rag_system.py       # RAG pipeline (embedding, retrieval, scoring)
├── pdf_to_vectors.py   # PDF ingestion & vectorisation
├── txt_to_vectors.py   # Text ingestion & vectorisation
├── templates/          # Jinja2 HTML templates
├── static/             # CSS, JS, images
├── uploads/            # Uploaded resumes (gitignored)
├── instance/           # SQLite database (gitignored)
├── requirements.txt
├── .env.example        # Template for environment variables
└── .gitignore
```

## Security

- **Never commit `.env`** — it is listed in `.gitignore`
- All API keys must be stored in `.env` only
- The `uploads/` and `instance/` directories are gitignored to protect user data

## License

MIT
