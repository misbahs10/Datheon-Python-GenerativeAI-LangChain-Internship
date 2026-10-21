# 🐍 Datheon Python + Generative AI & LangChain Internship

A collection of practical Python and Generative AI projects developed as part of the **Datheon Python + Generative AI & LangChain Internship**.

This repository showcases hands-on work with Large Language Models (LLMs), structured data extraction, Retrieval-Augmented Generation (RAG), semantic search, vector databases, and interactive AI applications built with Streamlit.

## 🚀 Projects

### 1. LLM JSON Extractor

**Folder:** `task-1-llm-json-extractor/`

A Python application that uses an LLM to extract structured information from unstructured text, such as resumes, and validate the result against a defined schema.

**Key features**
- Extracts structured candidate information from resume text.
- Uses Pydantic models for data validation.
- Produces structured JSON output.
- Handles candidate details, skills, and experience information.
- Uses environment variables for API configuration.

**Technologies:** Python, OpenAI-compatible API, Pydantic, python-dotenv.

### 2. RAG Document Q&A

**Folder:** `task-2-rag-document-qa/`

A document question-answering application that retrieves relevant passages from an uploaded PDF and uses a language model to answer questions grounded in the document.

**Key features**
- Uploads and extracts text from PDF documents.
- Splits extracted text into searchable chunks.
- Generates semantic embeddings using Sentence Transformers.
- Stores and retrieves vector representations with ChromaDB.
- Uses retrieved passages as context for answer generation.
- Displays relevant source passages and page numbers.
- Provides a clean Streamlit interface.

**Technologies:** Python, Streamlit, ChromaDB, Sentence Transformers, PyPDF, OpenRouter API.

### 3. Leaf AI Assistant — Streaming AI Chatbot

**Folder:** `task-4-streamlit-ai-chatbot/`

An interactive AI assistant that supports general conversational AI and PDF-based question answering through a unified Streamlit interface.

**Key features**
- General-purpose AI chat.
- PDF-based question answering using the Task 2 RAG pipeline.
- Streaming AI responses.
- Chat history managed with Streamlit session state.
- Retrieved PDF passages and page references.
- Sage green, cream, and white user interface.
- API credentials loaded from environment variables.

**Technologies:** Python, Streamlit, OpenAI Python SDK, OpenRouter API, ChromaDB, Sentence Transformers, PyPDF, python-dotenv.

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| LLM integration | OpenRouter API, OpenAI-compatible SDK |
| Data validation | Pydantic |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| PDF processing | PyPDF |
| Configuration | python-dotenv |
| Version control | Git and GitHub |

## 📁 Repository Structure

```text
Datheon-Python-GenerativeAI-LangChain-Internship/
│
├── task-1-llm-json-extractor/
│   ├── main.py
│   ├── schemas.py
│   └── README.md
│
├── task-2-rag-document-qa/
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── requirements.txt
│   └── README.md
│
├── task-4-streamlit-ai-chatbot/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
```

*Note: This structure highlights the main project files. Individual folders may contain additional supporting files.*

## ⚙️ Getting Started

### Prerequisites

- Python 3.12 or a compatible Python version.
- Git.
- An OpenRouter API key.
- Internet access for API requests and model downloads.

### 1. Clone the repository

```bash
git clone https://github.com/misbahs10/Datheon-Python-GenerativeAI-LangChain-Internship.git
cd Datheon-Python-GenerativeAI-LangChain-Internship
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Configure environment variables

Create a `.env` file in the repository root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Replace the placeholder with your own API key. Use a model available to your OpenRouter account and keep all credentials private.

### 4. Install project dependencies

Install the dependencies for the project you want to run.

**Task 1 — LLM JSON Extractor**

Install the dependencies listed in the Task 1 README and use its documented run command.

**Task 2 — RAG Document Q&A**

```powershell
python -m pip install -r task-2-rag-document-qa/requirements.txt
python -m streamlit run task-2-rag-document-qa/app.py
```

**Task 4 — Leaf AI Assistant**

```powershell
python -m pip install -r task-4-streamlit-ai-chatbot/requirements.txt
python -m streamlit run task-4-streamlit-ai-chatbot/app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`. If another Streamlit app is already running, Streamlit may select a different port.

## 🔄 How the RAG Pipeline Works

The document question-answering workflow follows these stages:

1. **Document ingestion:** Extract readable text and page numbers from the uploaded PDF.
2. **Chunking:** Divide the text into smaller passages for retrieval.
3. **Embedding generation:** Convert text chunks into vector embeddings.
4. **Vector storage:** Store embeddings and passage metadata in ChromaDB.
5. **Semantic retrieval:** Retrieve passages relevant to the user's question.
6. **Answer generation:** Provide retrieved passages to the language model as context.
7. **Source verification:** Display relevant source passages and page numbers alongside the answer.

## 🔐 Security and Privacy

- API credentials are loaded from the root `.env` file.
- The `.env` file and virtual environment should remain excluded through `.gitignore`.
- Never commit API keys, passwords, or other secrets to GitHub.
- Avoid uploading confidential documents to third-party AI services without authorization.
- Treat uploaded document content as untrusted input.
- Review API provider policies before submitting sensitive information.

## ⚠️ Limitations

- AI responses depend on the selected model, API availability, quota, and rate limits.
- PDF answers depend on the quality of extracted text and retrieved passages.
- Scanned PDFs may require OCR.
- Retrieval can return partially relevant passages.
- Local in-memory vector collections may need to be rebuilt when the application session restarts.
- A successful local test does not guarantee that every environment or model will behave identically.

## 🎯 Learning Outcomes

This project collection demonstrates practical experience with:

- Python application development.
- LLM API integration.
- Structured extraction and Pydantic validation.
- Embeddings and semantic search.
- Retrieval-Augmented Generation (RAG).
- Vector database integration.
- PDF text extraction and source references.
- Streaming responses and conversational session management.
- Building interactive AI applications with Streamlit.
- Git, GitHub, project documentation, and environment-variable security.

## 👩‍💻 Author

**Misbah Sajjad**

- **GitHub:** [misbahs10](https://github.com/misbahs10/Datheon-Python-GenerativeAI-LangChain-Internship)
- **LinkedIn:** [Misbah Sajjad](https://linkedin.com/in/misbahsajjad23)
- **Portfolio:** [AI & Data Science Portfolio](https://misbah-sajjad-ai-data-science-portfolio.ai.studio)

## 📌 Internship Information

- **Organization:** Datheon
- **Program:** Python + Generative AI & LangChain Internship
- **Repository:** [Datheon-Python-GenerativeAI-LangChain-Internship](https://github.com/misbahs10/Datheon-Python-GenerativeAI-LangChain-Internship)
- **Projects documented:** Task 1, Task 2, and Task 4

*This repository documents project work completed during the internship. Official internship completion and certificate issuance remain subject to Datheon's requirements and confirmation.*
