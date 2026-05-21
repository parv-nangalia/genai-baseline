import os
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from . import models


class DBConfig:
    db_connection = None
    _engine = None
    _SessionLocal = None

    @classmethod
    def get_connection(cls):
        # legacy psycopg2 connection (kept for compatibility)
        if cls.db_connection is None:
            print("Initializing DB connection (psycopg2)...")
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME", "genai")
            db_user = os.getenv("DB_USER", "postgres")
            db_pass = os.getenv("DB_PASS", "postgres")

            cls.db_connection = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_pass,
            )
        return cls.db_connection

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME", "genai")
            db_user = os.getenv("DB_USER", "postgres")
            db_pass = os.getenv("DB_PASS", "postgres")
            database_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            cls._engine = create_engine(database_url, pool_pre_ping=True)
            # create tables if they don't exist
            models.Base.metadata.create_all(bind=cls._engine)
        return cls._engine

    @classmethod
    def get_session(cls):
        if cls._SessionLocal is None:
            engine = cls.get_engine()
            cls._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return cls._SessionLocal()
