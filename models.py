# app/models.py
from sqlmodel import SQLModel, Field
from typing import Optional

class Resume(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    phone: str
    experience: str
    education: str
    skills: str
