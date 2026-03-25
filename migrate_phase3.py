from app import app, db
from models import User, Job, Candidate, Note, Interview

def upgrade_schema():
    with app.app_context():
        print("Creating Phase 3 Tables (Interview)...")
        db.create_all()
        print("Successfully migrated database for Phase 3!")

if __name__ == '__main__':
    upgrade_schema()
