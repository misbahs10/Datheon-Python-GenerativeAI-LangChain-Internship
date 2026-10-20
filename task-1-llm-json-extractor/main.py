
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schemas import ResumeData


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def extract_resume_data(resume_text: str) -> ResumeData:
    """Extract structured resume data using OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY missing. Check the root .env file."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract information from the supplied resume. "
                    "Return only a valid JSON object with these fields: "
                    "full_name (string), email (string or null), "
                    "skills (array of strings), "
                    "years_of_experience (number or null). "
                    "Do not invent missing information. "
                    "Use null for unknown email or experience, "
                    "and an empty array for unknown skills."
                ),
            },
            {
                "role": "user",
                "content": f"Resume:\n{resume_text}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("OpenRouter returned an empty response.")

    try:
        return ResumeData.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            f"Response does not match ResumeData: {error}"
        ) from error


if __name__ == "__main__":

    test_resumes = [
        {
            "test_name": "Test 1 - Experienced Candidate",
            "resume": """
            Name: Ayesha Khan
            Email: ayesha@example.com
            Skills: Python, SQL, Pandas, Machine Learning
            Experience: 2 years as a Junior Data Analyst
            """,
        },
        {
            "test_name": "Test 2 - Entry-Level Candidate",
            "resume": """
            Name: Ali Ahmed
            Email: ali.ahmed@example.com
            Skills: Python, Excel, Data Visualization
            Experience: 0 years
            """,
        },
    ]

    for test in test_resumes:
        print(f"\n{'=' * 45}")
        print(test["test_name"])
        print("=" * 45)

        try:
            result = extract_resume_data(test["resume"])
            print(result.model_dump_json(indent=2))

        except (ValueError, ValidationError) as error:
            print(f"Test failed: {error}")

        except Exception as error:
            print(f"API or unexpected error: {error}")
