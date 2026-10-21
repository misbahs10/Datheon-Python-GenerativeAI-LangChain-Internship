# 🌿 Leaf AI Assistant — Task 4

A conversational AI web application built with **Python, Streamlit, OpenRouter, and Retrieval-Augmented Generation (RAG)**. The application supports general AI conversations and document-based question answering with PDF source references.

## ✨ Features

- 💬 **General AI Chat:** Ask questions about Python, AI, machine learning, and other topics.
- 📄 **PDF Question Answering:** Upload a text-based PDF and ask questions about its content.
- 🔎 **Semantic Search:** Uses sentence-transformer embeddings and ChromaDB to retrieve relevant document passages.
- ⚡ **Streaming Responses:** Displays AI responses progressively as tokens arrive.
- 📚 **Source References:** Shows retrieved PDF passages and their page numbers.
- 🧠 **Chat History:** Maintains conversation history using Streamlit session state.
- 🎨 **Modern UI:** Minimal sage green, cream, and white interface.
- 🔐 **API Key Security:** Loads API credentials from environment variables instead of hardcoding them in source code.

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenAI Python SDK
- OpenRouter API
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`)
- PyPDF
- python-dotenv

## 📁 Project Structure

```text
task-4-streamlit-ai-chatbot/
├── app.py
├── requirements.txt
└── README.md
```

The application reuses the RAG pipeline from `task-2-rag-document-qa` in the parent internship repository.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/misbahs10/Datheon-Python-GenerativeAI-LangChain-Internship.git
cd Datheon-Python-GenerativeAI-LangChain-Internship
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the virtual environment already exists, activate it instead of creating it again.

### 3. Install dependencies

```powershell
python -m pip install -r task-4-streamlit-ai-chatbot/requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the repository root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Replace the placeholder with your own OpenRouter API key. Keep `.env` private and ensure it is excluded by `.gitignore`. Never commit API keys to GitHub.

### 5. Run the application

From the repository root:

```powershell
python -m streamlit run task-4-streamlit-ai-chatbot/app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 🚀 How to Use

### General AI Chat

1. Select **General AI Chat** in the sidebar.
2. Enter a question in the chat input.
3. Press Enter and view the streaming response.
4. Continue the conversation with follow-up questions.

### Chat with a PDF

1. Select **Chat with a PDF**.
2. Upload a text-based PDF document.
3. Wait for the document processing confirmation.
4. Ask a question about the document.
5. Read the answer and inspect the retrieved source passages and page numbers.

## 🔄 How It Works

### General AI Chat

The application sends the conversation history to the selected OpenRouter model and streams the generated response to the Streamlit interface.

### PDF Question Answering

1. **PDF extraction:** Extracts readable text and page information from the uploaded PDF.
2. **Chunking:** Splits document text into smaller passages.
3. **Embedding generation:** Converts passages into vector embeddings using Sentence Transformers.
4. **Retrieval:** Uses ChromaDB and relevance scoring to find passages related to the user's question.
5. **Answer generation:** Sends retrieved passages and the question to the language model.
6. **Source display:** Shows retrieved passages with page numbers to help users verify the answer.

## 🧪 Testing

The application was tested locally with the following scenarios:

- General AI questions and Python programming explanations.
- PDF upload and document processing.
- Questions about the main stages of Retrieval-Augmented Generation.
- Answers grounded in PDF passages with page citations.
- Display of retrieved source passages.
- Follow-up questions within the chat interface.

## ⚠️ Limitations

- Answers depend on the quality and content of the uploaded PDF.
- Scanned or image-only PDFs may require OCR.
- Retrieval may include passages that are only partially relevant.
- Response availability depends on the configured model, API quota, rate limits, and network connectivity.
- Uploaded documents and in-memory vector collections are intended for the current application session; persistent storage is not configured.

## 🔐 Security Notes

- Store API credentials in the root `.env` file.
- Never commit `.env`, API keys, or other secrets.
- Avoid uploading confidential documents to services unless authorized.
- Treat retrieved document content as untrusted input.

## 🎯 Internship Deliverable

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Task:** Task 4 — Streamlit AI Chatbot  
**Deliverable:** A locally running conversational AI web application featuring streaming responses, chat history, session state management, PDF-based RAG, and source references.

## 👩‍💻 Author

**Misbah Sajjad**

- GitHub: [@misbahs10](https://github.com/misbahs10/Datheon-Python-GenerativeAI-LangChain-Internship)
- Portfolio: [AI & Data Science Portfolio](https://misbah-sajjad-ai-data-science-portfolio.ai.studio)
- LinkedIn: [Misbah Sajjad](https://linkedin.com/in/misbahsajjad23)
