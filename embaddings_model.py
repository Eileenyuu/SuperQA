from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.embeddings.create(
  model="text-embedding-3-small",
  input="food",
  encoding_format="float"
)

#print(response)

print(response.data[0].embedding)