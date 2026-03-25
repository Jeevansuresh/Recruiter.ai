from app import app, db
from models import User, Job, Candidate, Note
from sqlalchemy import text
import traceback

def upgrade_schema():
    with app.app_context():
        # Create new Note table
        print("Creating newly defined tables (Note)...")
        db.create_all()
        
        # Manually alter Candidate table if needed
        print("Migrating candidate table...")
        try:
            db.session.execute(text("ALTER TABLE candidates ADD COLUMN status VARCHAR(50) DEFAULT 'Pending';"))
            db.session.commit()
            print("Successfully added 'status' column to candidates.")
        except Exception as e:
            if 'Duplicate column name' in str(e) or 'duplicate column' in str(e).lower() or 'operationalerror' in str(e).lower():
                print("Column 'status' already exists in candidates table.")
                db.session.rollback()
            else:
                print(f"Error altering table: {e}")
                traceback.print_exc()

if __name__ == '__main__':
    upgrade_schema()
    print("Migration complete!")
