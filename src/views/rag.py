from ..utility.urlParsing import parse_website
from ..utility.docParsing import process_document
from ..db.vectorIngestion import ingest_processed_chunks
from ..db.dbConfig import DBConfig
from ..utility.helper import get_doc_uuid


def ingestionView(target, targetType):
    db_connection = DBConfig.get_connection()
    if targetType=="url":
        ## logic to scrap using beautifulsoup
        try:
            response = parse_website(target,get_doc_uuid())
        except Exception as e:
            print("Error while parsing the website")
            raise
    else:
        ## logic to scrap using PDF OCR
        try:
            response = process_document(target, get_doc_uuid())
        except Exception as e:
            print("error while parsing the document")
            raise
    try:
        ingest_processed_chunks(db_connection, response)
    except Exception as e:
        print("Error while inserting the document into the db")
        raise

        

