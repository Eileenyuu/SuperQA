from milvus_database import *
from preprocess_text import *
from openai_models import extract_biomedical_terms, final_openai_output, csv_to_string
import asyncio  # Required for running async functions

async def main():
    query = input("Find papers about what: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return
    
    key_terms = extract_biomedical_terms(query)
    get_paper_information_paginated(key_terms, result_limit=50, required_count=20)
    connect_to_server()
    insert_into_collection("HealthTech")
    await search_similar_texts(key_terms)
    formatted_string = csv_to_string()
    print(final_openai_output(user_query=query, final_context=formatted_string))
    drop_collection("HealthTech")

# Run the async function
asyncio.run(main())