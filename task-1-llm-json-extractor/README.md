# LLM-Based Resume JSON Extractor

## Project Overview

This project uses a Large Language Model (LLM) to extract structured information from resume text and convert it into a validated JSON object. It uses OpenRouter to access an AI model and Pydantic to validate the extracted data.

## Features

- Extracts a candidate's full name and email address.
- Identifies technical and professional skills.
- Extracts years of experience when available.
- Returns structured JSON output.
- Validates the output using Pydantic.
- Handles missing information without intentionally inventing values.
- Uses environment variables to protect API credentials.

## Technologies Used

- Python
- OpenRouter API
- OpenAI Python SDK
- Pydantic
- python-dotenv
- JSON

## Project Structure

```text
task-1-llm-json-extractor/
├── main.py
├── schemas.py
└── README.md
```

The project also uses a `.env` file in the parent project directory to store the OpenRouter API key.

## Installation and Setup

### 1. Activate the virtual environment

From the main internship project directory, activate your existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install openai pydantic python-dotenv
```

### 3. Configure the API key

Create or update the `.env` file in the main project directory:

```env
OPENROUTER_API_KEY=your_actual_api_key
```

Replace the placeholder with your own OpenRouter API key. Never publish this key.

### 4. Run the project

Open a terminal in the `task-1-llm-json-extractor` directory and run:

```powershell
python main.py
```

## Example Input

```text
Name: Ayesha Khan
Email: ayesha@example.com
Skills: Python, SQL, Pandas, Machine Learning
Experience: 2 years as a Junior Data Analyst
```

## Example Output

```json
{
  "full_name": "Ayesha Khan",
  "email": "ayesha@example.com",
  "skills": [
    "Python",
    "SQL",
    "Pandas",
    "Machine Learning"
  ],
  "years_of_experience": 2.0
}
```

## Validation

Pydantic validates the extracted information against the `ResumeData` schema. The application reports an error if the returned JSON cannot be validated.

## Security

- Store API credentials in `.env`.
- Add `.env` and `.venv/` to `.gitignore`.
- Do not commit API keys or other secrets to GitHub.

## Learning Outcomes

- Working with an LLM API.
- Extracting structured information from unstructured text.
- Validating data with Pydantic.
- Handling JSON responses and API errors.
- Managing configuration through environment variables.

## Author

Misbah Sajjad

## Project Status

Core resume extraction and schema validation have been tested. Additional test cases and GitHub publication are pending.
