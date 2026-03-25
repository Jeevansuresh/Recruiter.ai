"""
init_db.py — Run this once after cloning to create the database tables.

Usage:
    python init_db.py

This is only needed if you want to initialise the DB without starting the
full app. Normally, running `python app.py` will auto-create all tables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from app import app
from models import db

with app.app_context():
    db.create_all()
    print("✅ Database tables created successfully.")
    print(f"   Location: {app.config['SQLALCHEMY_DATABASE_URI']}")
