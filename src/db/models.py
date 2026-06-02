from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import UserDefinedType
from sqlalchemy.orm import relationship

Base = declarative_base()


class Vector(UserDefinedType):
    def get_col_spec(self, **kwargs):
        return "VECTOR"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return "[" + ",".join(str(x) for x in value) + "]"
            return value
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                cleaned = value.strip("[]")
                if cleaned == "":
                    return []
                return [float(x) for x in cleaned.split(",")]
            return value
        return process


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    doc_id = Column(String, index=True)
    content = Column(Text)
    meta_data = Column(JSONB)

    embeddings = relationship("Embedding", back_populates="chunk")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String, ForeignKey("chunks.id"), index=True)
    model = Column(String, index=True)
    dimension = Column(Integer)
    embedding = Column(Vector)
    meta_data = Column(JSONB)

    chunk = relationship("Chunk", back_populates="embeddings")
