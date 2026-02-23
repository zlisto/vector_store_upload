# PDF to MongoDB Vector Store

Extracts text from a PDF, chunks it by tokens, embeds with Google Gemini, and stores in a MongoDB Atlas vector store for RAG (Retrieval Augmented Generation).

## Input

- **PDF file**: A text-based PDF (e.g. `data/harrypotter_complete_collection.pdf`)
  - Path is set in `PDF_PATH` (default: `data/harrypotter_complete_collection.pdf`)
  - Place your PDF in the `data/` folder or update the path in the script

## Environment Variables (.env)

Create a `.env` file in the project root with:

```
GEMINI_API_KEY=your_gemini_api_key_here
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/
```

| Variable       | Description                                      |
|----------------|--------------------------------------------------|
| `GEMINI_API_KEY` | Google AI API key for Gemini embedding model   |
| `MONGODB_URI`    | MongoDB Atlas connection string (with credentials) |

## Packages

| Package      | Version  | Purpose                          |
|--------------|----------|----------------------------------|
| PyMuPDF      | >=1.23.0 | PDF text extraction              |
| google-genai | >=1.0.0  | Gemini embedding API             |
| pymongo      | >=4.7.0  | MongoDB driver                   |
| python-dotenv| >=1.0.0  | Load .env variables              |
| tiktoken     | >=0.5.0  | Token-based chunking (cl100k)    |

## Setup & Run

```bash
pip install -r requirements.txt
python pdf_to_mongodb.py
```

## Output

- **MongoDB**: Database `rag_docs`, collection `harry_potter`
- **Documents**: Each chunk has `chunk_id`, `text`, `page_number` (zero-padded, e.g. `0001`), and `embedding` (768-dim vector)
- **Vector index**: `vector_index` for cosine similarity search

## Features

- Chunks by 500 tokens with 50-token overlap
- Skips chunks already in MongoDB (resumable)
- Retries on 429 rate limit with exponential backoff
- 2-second delay between batches (~750K TPM, under 1M limit)
- Prints total runtime at the end

---

## AI Prompt for Vibe Coder

Use this prompt to recreate this project with an AI coding assistant (e.g. Cursor, Vibe Coder):

---

**Build a Python script that:**

1. **Input**: Reads a PDF file (path configurable, default `data/harrypotter_complete_collection.pdf`). The PDF is a book with text only.

2. **Processing**:
   - Extract text from the PDF page by page, preserving page boundaries
   - Concatenate into one text string
   - Chunk the string by tokens: 500 tokens per chunk, 50-token overlap. Use tiktoken with `cl100k_base` encoding
   - For each chunk, track which page it came from (page_number metadata)

3. **Embedding**: Use Google Gemini API (`gemini-embedding-001`) to embed each chunk. Set output to 768 dimensions. Load API key from `.env` as `GEMINI_API_KEY`.

4. **Storage**: Store in MongoDB Atlas. Load `MONGODB_URI` from `.env`. Use database `rag_docs`, collection `harry_potter`. Create database and collection if they don't exist. Each document: `chunk_id` (SHA256 hash of page+text), `text`, `page_number` (zero-padded like `0001`, `0923` for sortable strings), `embedding`. Create a vector search index on the embedding field (768 dimensions, cosine similarity).

5. **Robustness**:
   - Skip chunks already in MongoDB (check by `chunk_id`) so the script is resumable
   - Embed and upload in batches of 50 chunks
   - Sleep 2 seconds between batches to stay under Gemini's 1M tokens/minute limit
   - On 429 rate limit, retry with exponential backoff (60s, 120s, 240s, etc., up to 5 attempts)
   - Print status for each batch (e.g. "Batch 3/70: embedding 50 chunks... done. Batch 3/70: uploaded 50 docs to MongoDB.")
   - Print total runtime at the end

6. **Packages**: PyMuPDF, google-genai, pymongo, python-dotenv, tiktoken. Include a `requirements.txt`.

7. **Output**: A runnable script that processes the PDF and populates the MongoDB vector store. No need to clear the collection on each run—skip existing chunks instead.
