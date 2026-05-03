"""
One-time migration script.
Run: python db/migrate.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import init_db, migrate_from_json, get_document_count

print("Initialising database...")
init_db()

print("Importing existing crawled documents...")
count = migrate_from_json(
    crawled_dir="data/crawled_distributed",
    delete_after=True       # deletes JSON files after successful import
)

print(f"\nDone. Documents in DB: {get_document_count()}")
print(f"Database location: data/search.db")