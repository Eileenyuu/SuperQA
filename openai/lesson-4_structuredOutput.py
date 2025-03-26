from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Expected output:
# {
#     'name': 'Science fair',
#     'date': 'Friday',
#     'location': 'Science Museum',
#     'attendees': ['Alice', 'Bob']
# }

# Structured Output mode
input_messages = [
    {
        "role": 'user',
        "content": "Tom and Bob are going to a science fair in London on Friday."
    }
]

response = client.responses.create(
    model = 'gpt-4o-mini',
    instructions = 'Extract the event information.',
    input = input_messages,
    #construct output data format using json schema
    text = {
        "format": {
           "type": "json_schema",
            "name": "calendar_event",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "date": {
                        "type": "string"
                    },
                    "location": {
                        "type": "string"
                    },
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["name", "date", "location", "attendees"],
                "additionalProperties": False
            },
            "strict": True #control model to strictly conform to everything set out in the schema
        }
    }
)

print(response.output_text)