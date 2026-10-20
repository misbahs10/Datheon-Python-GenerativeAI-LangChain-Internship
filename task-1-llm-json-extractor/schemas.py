
from pydantic import BaseModel, Field


class ResumeData(BaseModel):
    """Validated structure for extracted resume information."""

    full_name: str = Field(
        description="Candidate's full name"
    )

    email: str | None = Field(
        default=None,
        description="Candidate's email address, if available",
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Technical and professional skills",
    )

    years_of_experience: float | None = Field(
        default=None,
        ge=0,
        description="Years of experience, if known",
    )
