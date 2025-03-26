from openai import OpenAI
from llm_agent_templates import prompt_template, prompt_template2, prompt_template3
from apikeys import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def text_to_embedding(text):
    """
    Converts input text into an embedding using OpenAI's text-embedding-ada-002 model.
    
    Args:
        text (str): Input text to convert into an embedding.
    
    Returns:
        list: A list representing the text embedding.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in text_to_embedding: {e}")
        return None

def extract_biomedical_terms(query):
    """
    Extracts biomedical terms from a given query using GPT-4o.

    Args:
        query (str): The user-provided query.

    Returns:
        str: Extracted biomedical terms.
    """

    prompt = prompt_template.format(query=query)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in extract_biomedical_terms: {e}")
        return None


def final_openai_output(user_query, final_context):
    """
    Generates a final OpenAI response using GPT-4o based on user query and additional context.

    Args:
        user_query (str): The users original query.
        final_context (str): Additional contextual information.

    Returns:
        str: The generated response.
    """
    prompt = prompt_template2.format(query=user_query, context=final_context)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in final_openai_output: {e}")
        return None

def mcq_openai_output(user_query, final_context):
    """
    Generates a multiple-choice question (MCQ) response using GPT-4o.

    Args:
        user_query (str): The users query.
        final_context (str): Additional contextual information.

    Returns:
        str: The generated MCQ response.
    """

    prompt = prompt_template3.format(query=user_query, context=final_context)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1, # Only 1 token for the Lab Bench
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in mcq_openai_output: {e}")
        return None

# Model that I can use once API key comes back "gpt-4-turbo-2024-04-09"