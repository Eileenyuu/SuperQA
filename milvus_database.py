from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from nlp_class import EmbeddingLoader
from collections import defaultdict
import pandas as pd
import csv
import json

embedding_model = EmbeddingLoader()

def connect_to_server():
    connections.connect(host="localhost", port=29530)

def initialize_database_connections():
    """Initialize and return connections to collections."""
    collection = Collection(name="HealthTech")
    collection.load()
    return collection

# This collection is for user database and bot database, mainly initialising, run once.
def create_or_load_collection(collection_name="HealthTech"):
    try:
        # Ensure connection
        connect_to_server()
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            
            # Load collection
            collection.load()
            return collection

        # If collection doesn't exist, create it with schema and index
        
        # Define fields
        pk_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        paperID_field = FieldSchema(name="paperID", dtype=DataType.VARCHAR, max_length=256)
        title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512)
        abstract_text_field = FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=1024)
        embedding_field = FieldSchema(name="abstract_embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        year_field = FieldSchema(name="year", dtype=DataType.INT64)
        link_field = FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=512)
        
        # Create collection schema
        schema = CollectionSchema(
            fields=[pk_field, paperID_field, title_field, abstract_text_field, embedding_field, year_field, link_field], 
            description="HealthTech QA"
        )
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        collection.create_index(
            field_name="abstract_embedding", 
            index_params={
                "index_type": "IVF_FLAT", 
                "metric_type": "L2", 
                "params": {"nlist": 1024}
            }
        )
        check_index(collection_name, "abstract_embedding")
        return collection

    except Exception as e:
        print(f"Error in collection creation/loading: {e}")
        raise

def prepare_insertion_data(csv_file):

    df = pd.read_csv(csv_file)
    
    paper_ids = []
    titles = []
    abstracts = []
    embeddings = []
    years = []
    links = []

    # Process each row
    for _, row in df.iterrows():
        # Chunk the abstract
        chunks = chunk_message(row["abstract"])
        
        for chunk in chunks:
            # Generate embedding for the chunk
            chunk_embedding = embedding_model.process_text(chunk)

            # Append data to lists
            paper_ids.append(row["paperID"])
            titles.append(row["title"])
            abstracts.append(chunk)  # Store the chunk, not the whole abstract
            embeddings.append(chunk_embedding)
            years.append(int(row["year"]))
            links.append(row["link"])

    # Prepare data for insertion
    entities = [paper_ids, titles, abstracts, embeddings, years, links]

    return entities

def insert_into_collection(collection_name):
    """
    Inserts data into the collection.
    """
    collection = create_or_load_collection(collection_name)
    data = prepare_insertion_data(csv_file="outputs.csv")

    try:
        # Insert data into the partition
        collection.insert(data)
        collection.flush()
        return True

    except Exception as e:
        print(f"An error occurred during insertion: {e}")
        return False

async def search_similar_texts(query_text, top_k=5):
    """
    Performs similarity search on the embedding field and refines results
    based on timestamp if mentioned by the user, otherwise defaults to regular similarity search.
    """

    query_embedding = embedding_model.process_text(query_text)
    result2 = await search_with_similarity(query_embedding, top_k)
    return result2

async def search_with_similarity(query_embedding, top_k):
    """
    Performs regular similarity search without filtering based on timestamp.
    """
    try:
        # Ensure connection
        connections.connect(host="localhost", port=29530)
        
        # Load collections
        collection = Collection(name="HealthTech")
        collection.load()

        # Search for similar entries based on embedding
        query_results = collection.search(
            data=[query_embedding],
            anns_field="abstract_embedding",
            param={"metric_type": "L2", "params": {"nprobe": 50}},
            limit=top_k,
            output_fields=["title", "abstract", "year", "link"]
        )

        # Process and return results as usual
        if query_results:
            return await process_results(query_results)

        return None
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return None

async def process_results(query_results, output_file="output.csv"):
    """
    Processes search results, groups entries by title, year, and link,
    and writes the output to a CSV file without including paperID.
    """
    grouped_results = defaultdict(lambda: {"title": None, "year": None, "link": None, "abstracts": []})

    for result in query_results:
        for match in result:
            title = match.entity.get("title")
            year = match.entity.get("year")
            link = match.entity.get("link")
            abstract = match.entity.get("abstract")

            key = (title, year, link)

            if grouped_results[key]["title"] is None:  # Initialize only once
                grouped_results[key].update({"title": title, "year": year, "link": link})

            if abstract:
                grouped_results[key]["abstracts"].append(abstract)

    # Convert the grouped results into a list
    results_list = list(grouped_results.values())

    # Write to CSV file
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write header (NO paperID)
        writer.writerow(["title", "year", "link", "abstracts"])
        
        # Write rows (NO paperID)
        for entry in results_list:
            writer.writerow([
                entry["title"], 
                entry["year"], 
                entry["link"], 
                json.dumps(entry["abstracts"])  # Store abstracts as JSON
            ])
    
    return results_list

# Just for debugging purposes
def check_index(collection_name, field_name):
    collection = Collection(collection_name)
    indexes = collection.indexes
    for index in indexes:
        if index.field_name == field_name:
            return True
    return False

def chunk_message(abstract_text, chunk_size=250):
    words = abstract_text.split(" ")
    chunks = []
    current_chunk = ""

    for word in words:
        # Check if adding the next word exceeds the chunk size
        if len(current_chunk) + len(word) + 1 > chunk_size:
            # If it does, start a new chunk
            chunks.append(current_chunk)
            current_chunk = word
        else:
            # Otherwise, add the word to the current chunk
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def drop_collection(collection_name):
    try:
        # Connect to the Milvus server
        connect_to_server()
        
        # Create a collection object
        collection = Collection(name=collection_name)
        
        # Drop the collection
        collection.drop()
    
    except Exception as e:
        print(f"Error dropping collection: {e}")

def detailed_collection_diagnostics(collection_name):
    try:
        # Ensure connection
        connect_to_server()
        
        # Get the collection
        collection = Collection(name=collection_name)
        
        # Load collection
        collection.load()
        collection.flush()
        
        # Comprehensive diagnostics
        print("\n--- Collection Diagnostics ---")
        print(f"Collection Name: {collection_name}")
        print(f"Total Entities: {collection.num_entities}")
    
        try:
            # Query a few entities to verify collection functionality
            query_result = collection.query(
                expr="id >= 0",  # Adjust expression as needed
                output_fields=["id", "paperID", "title", "abstract", "abstract_embedding", "year"], 
                limit=10
            )
            for entity in query_result:
                print(f"""
                paperID: {entity['paperID']}
                Title: {entity['title']}
                Abstract text: {entity['abstract']}
                Embedding: {entity['abstract_embedding'][:5]}... (truncated)
                Year: {entity['year']}
                """)
        except Exception as query_error:
            print(f"Query Error for partition: {query_error}")

    except Exception as e:
        print(f"Diagnostic Error: {e}")