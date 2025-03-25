from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
load_dotenv()

from pydantic import BaseModel, ConfigDict
from typing import List
import json
import os

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Define a prompt template
# This template instructs the LLM to find the key words or phrases
# from a user question and return them in a structured format.


# Read CSV File
file_name = f'outputsss.csv'
csv_file_path = Path('data')/ file_name
df = pd.read_csv(csv_file_path)

# Create the Pydantic class for papers
class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid") # same function as "addtionalProperties": False

    paperID: str
    title: str
    year: str
    abstract: str
    link: str

# Create the Pydantic class for paper list
class PaperList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    papers: List[Paper] # API post paper list

# Construct JSON Schema
schema = PaperList.model_json_schema()

# Construct input messages (link paper info in CSV)
content = "\n\n".join([
    f"The paper titled '{row['title']}' was published in {row['year']}. "
    f"It discusses {row['abstract']}. Read more: {row['link']}"
    for _, row in df.iterrows()
])

def prompt_build():
    prompt= f''


# Generate API input
input_messages = [
    {
        "role": 'user',
        "content": content
    }
]

client = OpenAI()

# Structured Output mode
response = client.responses.create(
    model = 'gpt-4o-mini',
    instructions = '''
    Extract all research paper details (paper ID, title, year, abstract, link)
    from the given text.
    ''',
    input = input_messages,
    #construct output data format using json schema
    text = {
        "format": {
           "type": "json_schema",
            "name": "research_papers",
            "schema": schema,
            "strict": True #control model to strictly conform to everything set out in the schema
        }
    }
)

# Pretty print the response as a formatted JSON string
# formatted_response = json.dumps(response.output_text, indent=4)
# print(formatted_response)

# print(response)
print(response.output_text)