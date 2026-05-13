"""
UNIFIED URL INGESTION FORMAT
----------------------------

Returns EXACT SAME structure as PDF/DOC ingestion.

FINAL OUTPUT FORMAT:

[
    {
        "chunk_id": "...",
        "doc_id": "...",
        "text": "...",
        "metadata": {...},
        "embeddings": {
            "openai": {
                "model": "...",
                "vector": [...]
            },
            "hf": {
                "model": "...",
                "vector": [...]
            }
        }
    }
]

This keeps:
- URL ingestion
- PDF ingestion
- DOC ingestion

100% consistent.

Your DB insertion layer can now be fully generic.
"""

import re
import requests

from bs4 import BeautifulSoup

from urllib.parse import urljoin

from uuid import uuid4

from .helper import (
    get_openai_embedding,
    get_hf_embedding_textual
)


# =========================================================
# CONFIG
# =========================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

MIN_CHUNK_LENGTH = 120

OPENAI_MODEL = "text-embedding-3-small"

HF_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# =========================================================
# FETCH HTML
# =========================================================

def fetch_html(url: str) -> str:

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return response.text



# =========================================================
# FAKE EMBEDDING UTILS
# ========================================================
import numpy as np
def fake_openai_vector(seed: int = 42):
    np.random.seed(seed)
    return np.random.normal(0, 1, 1536).tolist()



# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text: str) -> str:

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# REMOVE NOISE
# =========================================================

def remove_noise(soup: BeautifulSoup):

    noisy_tags = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "svg",
        "form"
    ]

    for tag in noisy_tags:

        for element in soup.find_all(tag):

            element.decompose()


# =========================================================
# EXTRACT TITLE
# =========================================================

def extract_title(soup: BeautifulSoup) -> str:

    h1 = soup.find("h1")

    if h1:
        return clean_text(h1.get_text())

    if soup.title:
        return clean_text(soup.title.get_text())

    return "Untitled"


# =========================================================
# EXTRACT MAIN CONTENT
# =========================================================

def extract_main_content(soup):

    candidates = [

        soup.find("article"),

        soup.find("main"),

        soup.find("div", class_="content"),

        soup.find("div", class_="main")
    ]

    for candidate in candidates:

        if candidate:
            return candidate

    return soup.body


# =========================================================
# SEMANTIC CHUNKING
# =========================================================

def chunk_by_headings(
    main_content,
    page_title,
    page_url
):

    chunks = []

    current_heading = "Introduction"

    current_text = []

    elements = main_content.find_all([
        "h1",
        "h2",
        "h3",
        "p",
        "li",
        "pre",
        "code"
    ])

    for element in elements:

        tag_name = element.name

        text = clean_text(
            element.get_text(" ")
        )

        if not text:
            continue

        # -------------------------------------------------
        # NEW SECTION
        # -------------------------------------------------

        if tag_name in ["h1", "h2", "h3"]:

            # SAVE PREVIOUS CHUNK
            if current_text:

                combined_text = "\n".join(current_text)

                if len(combined_text) >= MIN_CHUNK_LENGTH:

                    chunks.append({

                        "chunk_id": str(uuid4()),

                        "text": combined_text,

                        "metadata": {

                            "url": page_url,

                            "title": page_title,

                            "section": current_heading,

                            "content_type": "documentation"
                        }
                    })

            current_heading = text

            current_text = []

        else:
            current_text.append(text)

    # -----------------------------------------------------
    # FINAL CHUNK
    # -----------------------------------------------------

    if current_text:

        combined_text = "\n".join(current_text)

        if len(combined_text) >= MIN_CHUNK_LENGTH:

            chunks.append({

                "chunk_id": str(uuid4()),

                "text": combined_text,

                "metadata": {

                    "url": page_url,

                    "title": page_title,

                    "section": current_heading,

                    "content_type": "documentation"
                }
            })

    return chunks


# =========================================================
# OPTIONAL INTERNAL LINK EXTRACTION
# =========================================================

def extract_internal_links(
    soup,
    base_url
):

    links = set()

    for a in soup.find_all("a", href=True):

        href = a["href"]

        absolute_url = urljoin(base_url, href)

        if absolute_url.startswith(base_url):

            links.add(absolute_url)

    return list(links)


# =========================================================
# MAIN URL INGESTION PIPELINE
# =========================================================

def parse_website(
    url: str,
    doc_id: str
):

    """
    RETURNS:

    [
        {
            "chunk_id": "...",
            "doc_id": "...",
            "text": "...",
            "metadata": {...},
            "embeddings": {
                "openai": {...},
                "hf": {...}
            }
        }
    ]
    """

    # -----------------------------------------------------
    # FETCH + PARSE
    # -----------------------------------------------------

    html = fetch_html(url)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    remove_noise(soup)

    # -----------------------------------------------------
    # EXTRACT CONTENT
    # -----------------------------------------------------

    title = extract_title(soup)

    main_content = extract_main_content(soup)

    # -----------------------------------------------------
    # CHUNK CONTENT
    # -----------------------------------------------------

    chunks = chunk_by_headings(
        main_content,
        title,
        url
    )

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    processed_chunks = []

    for chunk in chunks:

        chunk_text = chunk["text"]

        metadata = chunk["metadata"]

        chunk_id = chunk["chunk_id"]

        # -------------------------------------------------
        # EMBEDDINGS
        # -------------------------------------------------

        # openai_embedding = get_openai_embedding(
        #     chunk_text
        # )

        openai_embedding = fake_openai_vector(seed=1)

        hf_embedding = get_hf_embedding_textual(
            chunk_text
        )

        # -------------------------------------------------
        # FINAL STRUCTURE
        # -------------------------------------------------
        processed_chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": chunk_text,
            "metadata": metadata,
            "embeddings": {
                "openai": {
                    "model": OPENAI_MODEL,
                    "vector": openai_embedding
                },
                "hf": {
                    "model": HF_MODEL_NAME,
                    "vector": hf_embedding
                }
            }
        })

    return processed_chunks