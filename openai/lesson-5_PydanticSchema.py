from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, ConfigDict
from typing import List

# Create the Pydantic class
class Event(BaseModel):
    model_config = ConfigDict(extra="forbid") # same function as "addtionalProperties": False

    name: str
    date: str
    location: str
    attendees: List[str]

schema = Event.model_json_schema()


#Initial messages
input_messages = [
    {
        "role": 'user',
        "content": "Tom and Bob are going to a science fair in London on Friday."
    }
]

client = OpenAI()

# Expected output:
# {
#     'name': 'Science fair',
#     'date': 'Friday',
#     'location': 'Science Museum',
#     'attendees': ['Alice', 'Bob']
# }

# Structured Output mode
response = client.responses.create(
    model = 'gpt-4o-mini',
    instructions = 'Extract the event information.',
    input = input_messages,
    #construct output data format using json schema
    text = {
        "format": {
           "type": "json_schema",
            "name": "calendar_event",
            "schema": schema,
            "strict": True #control model to strictly conform to everything set out in the schema
        }
    }
)

print(response.output_text)