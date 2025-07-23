from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL") 

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db():
    """Returns a database session and ensures it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Creates all defined tables in the database."""
    Base.metadata.create_all(bind=engine)
