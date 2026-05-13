import psycopg2

class DBConfig:
    db_connection = None

    @classmethod
    def get_connection(cls):
        if cls.db_connection is None:
            print("Initializing DB connection...")
            cls.db_connection = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/genai")
        return cls.db_connection
