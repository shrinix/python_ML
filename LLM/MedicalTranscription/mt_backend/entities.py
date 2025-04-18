from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Entities(BaseModel):
    """
    Identify and capture information about entities from text
    """

    names: List[str] = Field(
        description=
            "All the objects, persons, organizations, companies, medical concepts, medicines, medical conditins, or business entities that appear in the text",
    )