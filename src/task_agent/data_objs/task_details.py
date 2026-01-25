from typing import List
from pydantic import BaseModel, Field, field_validator


class TODOs_Output(BaseModel):
    output: str = ""
    model_used: str = ""
    execution_time: str = ""


class TODO_details(BaseModel):
    todo_id: str | None = None
    todo_name: str
    todo_description: str
    todo_completed: bool = False
    output: TODOs_Output | None = None


class TODOs(BaseModel):
    todo_list: List[TODO_details]
