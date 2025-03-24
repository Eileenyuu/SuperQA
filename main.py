from milvus_database import *
from preprocess_text import *
import asyncio  # Required for running async functions

async def main():
    query = input("Find papers about what: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return
    
    get_paper_information_paginated(query)

    connect_to_server()
    insert_into_collection("HealthTech")
    detailed_collection_diagnostics("HealthTech")
    results = await search_similar_texts(query)
    print(results)
    drop_collection("HealthTech")
    

# Run the async function
asyncio.run(main())