from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db, Job, Candidate, Note, Interview

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard_bp.route('/')
@login_required
def home():
    jobs = Job.query.filter_by(user_id=current_user.id).order_by(Job.created_at.desc()).all()
    return render_template('dashboard.html', jobs=jobs)

@dashboard_bp.route('/job/create', methods=['POST'])
@login_required
def create_job():
    title = request.form.get('title')
    description = request.form.get('description')
    if title and description:
        new_job = Job(user_id=current_user.id, title=title, description=description)
        db.session.add(new_job)
        db.session.commit()
        flash('Job created successfully!', 'success')
    else:
        flash('Title and description are required.', 'error')
    return redirect(url_for('dashboard.home'))

@dashboard_bp.route('/job/<int:job_id>')
@login_required
def job_details(job_id):
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        from flask import abort
        abort(403)
        
    # Use session.get instead of session.pop so the list persists
    # across redirects from shortlist/reject actions
    from flask import session
    recent_ids = session.get('recent_candidates', [])
    recent_candidates = []
    if recent_ids:
        recent_candidates = Candidate.query.filter(Candidate.id.in_(recent_ids)).order_by(Candidate.match_score.desc()).all()
        
    candidates = Candidate.query.filter_by(job_id=job.id).order_by(Candidate.match_score.desc()).all()
    return render_template('job_details.html', job=job, candidates=candidates, recent_candidates=recent_candidates)

@dashboard_bp.route('/candidate/<int:id>/status', methods=['POST'])
@login_required
def update_status(id):
    candidate = Candidate.query.get_or_404(id)
    if candidate.job.user_id != current_user.id:
        from flask import abort
        abort(403)
    
    new_status = request.form.get('status')
    if new_status in ['Pending', 'Shortlisted', 'Rejected']:
        candidate.status = new_status
        db.session.commit()
        flash(f'Candidate marked as {new_status}', 'success')
    
    # Redirect back to the job page so we keep the session context alive
    return redirect(url_for('dashboard.job_details', job_id=candidate.job_id))

@dashboard_bp.route('/candidates')
@login_required
def all_candidates():
    # Show all candidates across all jobs
    jobs = Job.query.filter_by(user_id=current_user.id).all()
    job_ids = [j.id for j in jobs]
    candidates = Candidate.query.filter(Candidate.job_id.in_(job_ids)).order_by(Candidate.match_score.desc()).all()
    return render_template('candidates.html', candidates=candidates)

@dashboard_bp.route('/workspace', methods=['GET', 'POST'])
@login_required
def workspace():
    notes = Note.query.filter_by(user_id=current_user.id).order_by(Note.updated_at.desc()).all()
    active_note = None
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'create':
            n = Note(user_id=current_user.id, title="Untitled Note", content="")
            db.session.add(n)
            db.session.commit()
            return redirect(url_for('dashboard.workspace', note_id=n.id))
        elif action == 'generate_jd':
            role_prompt = request.form.get('role_prompt')
            import google.generativeai as genai
            try:
                model = genai.GenerativeModel("gemini-2.5-pro")
                prompt = f"""Write a highly professional and detailed job description for: {role_prompt}

CRITICAL INSTRUCTIONS:
- Return ONLY the raw Markdown content. No preamble, no intro phrases like "Of course", "Here is", "Certainly", etc.
- Start IMMEDIATELY with the job title as a Markdown heading (# Title).
- Do not wrap in code blocks.
- Format in clean, structured Markdown with sections."""
                content = model.generate_content(prompt).text.strip()
                # Strip any lingering preamble lines before the first heading
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('#'):
                        content = '\n'.join(lines[i:])
                        break
                n = Note(user_id=current_user.id, title=f"JD Blueprint: {role_prompt}", content=content)
                db.session.add(n)
                db.session.commit()
                return redirect(url_for('dashboard.workspace', note_id=n.id))
            except Exception as e:
                flash(f"AI Generation failed: {str(e)}", "error")
                return redirect(url_for('dashboard.workspace'))
        elif action == 'save':
            note_id = request.form.get('note_id')
            n = Note.query.get(note_id)
            if n and n.user_id == current_user.id:
                n.title = request.form.get('title', 'Untitled Note')
                n.content = request.form.get('content', '')
                db.session.commit()
            return redirect(url_for('dashboard.workspace', note_id=note_id))
        elif action == 'delete':
            note_id = request.form.get('note_id')
            n = Note.query.get(note_id)
            if n and n.user_id == current_user.id:
                db.session.delete(n)
                db.session.commit()
            return redirect(url_for('dashboard.workspace'))
            
    note_id = request.args.get('note_id')
    if note_id:
        active_note = Note.query.filter_by(id=note_id, user_id=current_user.id).first()
    elif notes:
        active_note = notes[0]
        
    return render_template('workspace.html', notes=notes, active_note=active_note)

@dashboard_bp.route('/interviews', methods=['GET', 'POST'])
@login_required
def interviews():
    if request.method == 'POST':
        candidate_id = request.form.get('candidate_id')
        new_stage = request.form.get('stage')
        c = Candidate.query.get(candidate_id)
        if c and c.job.user_id == current_user.id:
            if not c.interview:
                i = Interview(candidate_id=c.id, stage=new_stage)
                db.session.add(i)
            else:
                c.interview.stage = new_stage
            db.session.commit()
        return redirect(url_for('dashboard.interviews'))
        
    jobs = Job.query.filter_by(user_id=current_user.id).all()
    job_ids = [j.id for j in jobs]
    
    # We want to show candidates that are Shortlisted or have an Interview record
    candidates = Candidate.query.filter(Candidate.job_id.in_(job_ids)).filter((Candidate.status == 'Shortlisted') | (Candidate.interview.has())).all()
    return render_template('interviews.html', candidates=candidates)

@dashboard_bp.route('/outreach', methods=['GET', 'POST'])
@login_required
def outreach():
    jobs = Job.query.filter_by(user_id=current_user.id).all()
    job_ids = [j.id for j in jobs]
    candidates = Candidate.query.filter(Candidate.job_id.in_(job_ids)).filter(Candidate.status != 'Pending').all()
    
    selected_candidate = None
    draft_email = ""
    
    if request.method == 'POST':
        candidate_id = request.form.get('candidate_id')
        action_type = request.form.get('action_type') # 'invite' or 'reject'
        selected_candidate = Candidate.query.get(candidate_id)
        
        if selected_candidate and selected_candidate.job.user_id == current_user.id:
            import google.generativeai as genai
            model = genai.GenerativeModel("gemini-2.5-pro")
            prompt = f"Draft a professional {action_type} email for the candidate applying to {selected_candidate.job.title}. AI Match Info: {selected_candidate.ai_analysis}. If inviting to interview, be warm and mention their strengths. If rejecting, be polite and specifically mention missing skills as constructive feedback. Keep it under 200 words. Do not use placeholders."
            try:
                draft_email = model.generate_content(prompt).text.strip()
            except Exception as e:
                draft_email = f"Error generating email: {str(e)}"
                
    return render_template('outreach.html', candidates=candidates, selected=selected_candidate, draft=draft_email)

@dashboard_bp.route('/analytics')
@login_required
def analytics():
    jobs = Job.query.filter_by(user_id=current_user.id).all()
    job_ids = [j.id for j in jobs]
    
    if not job_ids:
        candidates = []
    else:
        candidates = Candidate.query.filter(Candidate.job_id.in_(job_ids)).all()
    
    total_candidates = len(candidates)
    avg_score = sum([c.match_score for c in candidates if c.match_score]) / max(total_candidates, 1)
    
    # Compute missing skills aggregation
    missing_skills = {}
    for c in candidates:
        analysis = c.ai_analysis
        if analysis and isinstance(analysis, dict) and 'missing_critical_skills' in analysis:
            skills = analysis.get('missing_critical_skills', [])
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, str) and len(skill) > 2:
                        missing_skills[skill] = missing_skills.get(skill, 0) + 1
                    
    top_missing = sorted(missing_skills.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return render_template('analytics.html', total=total_candidates, avg_score=avg_score, top_missing=top_missing)
