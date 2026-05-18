# GenAI Baseline

A FastAPI-based generative AI baseline project with support for OpenAI and Google Gemini integration.

## Docker

### Build

```bash
docker build -t genai-baseline .
```

### Run

#### With Docker Compose

```bash
docker compose up --build
```

This starts the FastAPI app and a PostgreSQL database with the `pgvector` extension enabled.

#### Without Compose

Use your local environment file or Docker env file support.

```bash
docker run --rm -p 8000:8000 --env-file src/.env genai-baseline
```

If you prefer to mount the `.env` file directly:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/src/.env:/app/src/.env" genai-baseline
```

On Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/src/.env:/app/src/.env" genai-baseline
```

### Health check

Open `http://localhost:8000/health`

## Local development

1. Create a `.env` file in `src/` with:

```text
OPENAI_API_KEY="your_openai_key"
HUGGINGFACE_API_KEY="your_huggingface_key"
GOOGLE_API_KEY="your_google_key"
```

If you are using a local PostgreSQL database, also add:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=genai
DB_USER=postgres
DB_PASS=postgres
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Notes

- The FastAPI app is defined in `src/main.py` and loads environment variables from `src/.env`.
- Do not bake secrets into the image for production builds. Use `--env-file` or runtime environment variables instead.
