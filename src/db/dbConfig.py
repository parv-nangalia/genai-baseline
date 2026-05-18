import os
import psycopg2

class DBConfig:
    db_connection = None

    @classmethod
    def get_connection(cls):
        if cls.db_connection is None:
            print("Initializing DB connection...")
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
