"""Extract PDF text, chunk it, embed with Gemini, and store in MongoDB vector store."""

import hashlib
import os
import time
import fitz  # PyMuPDF
import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

PDF_PATH = "data/harrypotter_complete_collection.pdf"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
EMBEDDING_DIMENSIONS = 768
EMBED_BATCH_SIZE = 50
# Stay under 1M TPM: 50 chunks × 500 tokens = 25K/batch. At 2s delay = 30 batches/min = 750K TPM.
BATCH_DELAY_SEC = 2
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 60

DB_NAME = "rag_docs"
COLLECTION_NAME = "harry_potter"
VECTOR_INDEX_NAME = "vector_index"


def extract_pdf_to_string_with_page_map(pdf_path: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Extract PDF to a single text string and a page map.
    Page map: list of (token_start, token_end) per page (0-based page index).
    Returns (full_text, token_boundaries_per_page).
    """
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_texts.append(page.get_text())
    doc.close()

    full_text = "\n".join(page_texts)
    enc = tiktoken.get_encoding("cl100k_base")

    # Build token boundaries per page (matches full_text tokenization)
    page_boundaries: list[tuple[int, int]] = []
    offset = 0
    for i, pt in enumerate(page_texts):
        page_tokens = enc.encode(pt)
        start = offset
        end = offset + len(page_tokens)
        page_boundaries.append((start, end))
        offset = end
        if i < len(page_texts) - 1:
            offset += len(enc.encode("\n"))

    return full_text, page_boundaries


def chunk_string_by_tokens(
    text: str,
    page_boundaries: list[tuple[int, int]],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[tuple[str, int]]:
    """
    Chunk the full text string by tokens with overlap.
    Returns list of (chunk_text, page_number). Page number is 1-based.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    step = chunk_size - overlap
    chunks: list[tuple[str, int]] = []

    def token_to_page(token_idx: int) -> int:
        for page_idx, (start, end) in enumerate(page_boundaries):
            if start <= token_idx < end:
                return page_idx + 1
        return len(page_boundaries)

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        page_num = token_to_page(start)
        chunks.append((chunk_text.strip(), page_num))
        if end >= len(tokens):
            break
        start += step

    return chunks


def chunk_id(chunk_text: str, page_num: int) -> str:
    """Unique ID for a chunk (for skip-if-exists check)."""
    key = f"{str(page_num).zfill(4)}::{chunk_text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def embed_and_upload_batches(
    client: genai.Client,
    chunks: list[tuple[str, int]],
    collection,
) -> None:
    """Embed chunks with Gemini, upload to MongoDB periodically. Skips chunks already uploaded."""
    existing_ids = set(collection.distinct("chunk_id"))
    to_process = [(c, chunk_id(c[0], c[1])) for c in chunks if chunk_id(c[0], c[1]) not in existing_ids]
    skipped = len(chunks) - len(to_process)
    if skipped:
        print(f"   Skipping {skipped:,} chunks already in MongoDB.")
    if not to_process:
        print("   All chunks already uploaded.")
        return

    total = len(to_process)
    num_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch_num = (i // EMBED_BATCH_SIZE) + 1
        batch_items = to_process[i : i + EMBED_BATCH_SIZE]
        batch_chunks = [item[0] for item in batch_items]
        batch_texts = [c[0] for c in batch_chunks]

        print(f"   Batch {batch_num}/{num_batches}: embedding {len(batch_texts)} chunks...", end=" ", flush=True)
        for attempt in range(MAX_RETRIES):
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch_texts,
                    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
                )
                break
            except ClientError as e:
                if e.code == 429 and attempt < MAX_RETRIES - 1:
                    wait = INITIAL_BACKOFF_SEC * (2**attempt)
                    print(f"rate limited, waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...", flush=True)
                    time.sleep(wait)
                else:
                    raise
        embeddings = [list(emb.values) for emb in result.embeddings]
        print("done.", flush=True)

        docs = [
            {
                "chunk_id": cid,
                "text": chunk_text,
                "page_number": str(page_num).zfill(4),
                "embedding": emb,
            }
            for ((chunk_text, page_num), cid), emb in zip(batch_items, embeddings)
        ]
        collection.insert_many(docs)
        print(f"   Batch {batch_num}/{num_batches}: uploaded {len(docs)} docs to MongoDB.", flush=True)

        if i + EMBED_BATCH_SIZE < total:
            time.sleep(BATCH_DELAY_SEC)


def main():
    start = time.perf_counter()
    api_key = os.getenv("GEMINI_API_KEY")
    mongodb_uri = os.getenv("MONGODB_URI")
    if not api_key or not mongodb_uri:
        raise ValueError("GEMINI_API_KEY and MONGODB_URI must be set in .env")

    print("1. Extracting PDF to text string...")
    full_text, page_boundaries = extract_pdf_to_string_with_page_map(PDF_PATH)
    print(f"   Extracted {len(full_text):,} chars from {len(page_boundaries)} pages")

    print("2. Chunking string by tokens...")
    chunks = chunk_string_by_tokens(full_text, page_boundaries)
    print(f"   Created {len(chunks):,} chunks ({CHUNK_SIZE_TOKENS} tokens, {CHUNK_OVERLAP_TOKENS} overlap)")

    print("3. Connecting to MongoDB...")
    mongo_client = MongoClient(mongodb_uri)
    db = mongo_client[DB_NAME]
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"   Created database '{DB_NAME}' and collection '{COLLECTION_NAME}'")
    collection = db[COLLECTION_NAME]

    print("4. Creating vector search index (if not exists)...")
    existing = list(collection.list_search_indexes())
    if not any(idx.get("name") == VECTOR_INDEX_NAME for idx in existing):
        collection.create_search_index(
            model=SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": EMBEDDING_DIMENSIONS,
                            "similarity": "cosine",
                        }
                    ]
                },
                name=VECTOR_INDEX_NAME,
                type="vectorSearch",
            )
        )
        print("   Index created")
    else:
        print("   Index already exists")

    print(f"5. Embedding and uploading to MongoDB ({BATCH_DELAY_SEC}s between batches, retries on 429)...")
    client = genai.Client(api_key=api_key)
    embed_and_upload_batches(client, chunks, collection)

    elapsed = time.perf_counter() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"Done! {collection.count_documents({}):,} documents in {DB_NAME}.{COLLECTION_NAME}")
    print(f"Total time: {mins}m {secs}s ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
