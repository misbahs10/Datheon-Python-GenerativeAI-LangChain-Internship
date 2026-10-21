
# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that answers questions from uploaded PDF documents using semantic search and a large language model (LLM).

## Features

- PDF text extraction with page numbers
- Text chunking
- Sentence Transformers embeddings
- ChromaDB vector search
- OpenRouter-powered question answering
- Source page references
- Streamlit user interface
- Answers grounded in retrieved PDF passages

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- OpenAI Python SDK
- OpenRouter API

## Setup

Run the following commands from the repository root.

Install dependencies:

```bash
python -m pip install -r task-2-rag-document-qa/requirements.txt
```

Create a `.env` file in the repository root and add your API key:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Never commit your real API key to GitHub.

## Run

From the repository root:

```bash
python -m streamlit run task-2-rag-document-qa/app.py
```

Upload a PDF, wait for processing, and ask questions about its contents.

## Limitations

- Scanned or image-only PDFs may require OCR.
- The embedding model may download on first use.
- The vector collection is stored in memory and rebuilt when a new PDF is processed.
- Answer quality depends on document text, retrieval relevance, and LLM availability.
- Retrieved source passages may include irrelevant text; retrieval quality can be improved further.
- An OpenRouter model or API quota issue may prevent answer generation.
