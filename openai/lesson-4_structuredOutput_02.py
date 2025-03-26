from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()

client = OpenAI()

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("menu.png")

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
        'content': [
            {
                "type": "input_text",
                "text": "Please extract all menu items."
            },
            #passing image as filepath
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        ]
    }
]

response = client.responses.create(
    model = 'gpt-4o-mini',
    input = input_messages,
    #construct output data format using json schema
    text = {
        "format": {
           "type": "json_schema",
            "name": "menu_items",
            "schema": {
                "type": "object",
                "properties": {
                    "menu_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_name": {
                                    "type": "string"
                                },
                                "item_price": {
                                    "type": "number"
                                }
                            },
                            "required": ["item_name", "item_price"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["menu_items"],
                "additionalProperties": False
            },
            "strict": True #control model to strictly conform to everything set out in the schema
        }
    }
)

print(response.output_text)