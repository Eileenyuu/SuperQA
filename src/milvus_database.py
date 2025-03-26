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

# Initialize the embedding model using the sentence transformer MiniLM-L6-v2
embedding_model = EmbeddingLoader()

def connect_to_server():
    """
    Establish a connection to the Milvus server.
    
    Milvus is a vector database used to store and search high-dimensional embeddings.
    This function connects to a Milvus instance running on localhost at port 29530.
    """
    connections.connect(host="localhost", port=29530)

def initialize_database_connections():
    """
    Initialize and return a connection to the 'HealthTech' collection in Milvus.

    The function:
    1. Connects to the 'HealthTech' collection.
    2. Loads the collection into memory for querying.

    Returns:
        Collection: The loaded Milvus collection.
    """
    collection = Collection(name="HealthTech")  # Connect to the HealthTech collection
    collection.load()  # Load the collection into memory
    return collection

def create_or_load_collection(collection_name="HealthTech"):
    """
    Create or load a Milvus collection for relevant scholarly articles.

    This function:
    1. Ensures a connection to the Milvus server.
    2. Checks if the collection already exists.
    3. If the collection exists, it is loaded into memory.
    4. If the collection does not exist, it is created with a predefined schema.
    5. An index is created on the "abstract_embedding" field for efficient searches.

    Args:
        collection_name (str): Name of the collection to create or load.

    Returns:
        Collection: The loaded or newly created Milvus collection.
    """
    try:
        # Ensure connection to Milvus server
        connect_to_server()
        
        # Check if collection already exists
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            collection.load()  # Load collection into memory
            return collection

        # Define schema fields for the collection
        pk_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        paperID_field = FieldSchema(name="paperID", dtype=DataType.VARCHAR, max_length=256)
        title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512)
        abstract_text_field = FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=1024)
        embedding_field = FieldSchema(name="abstract_embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        year_field = FieldSchema(name="year", dtype=DataType.INT64)
        link_field = FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=1024)
        
        # Create collection schema
        schema = CollectionSchema(
            fields=[pk_field, paperID_field, title_field, abstract_text_field, embedding_field, year_field, link_field], 
            description="HealthTech QA"
        )
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index on the embedding field for efficient similarity search
        collection.create_index(
            field_name="abstract_embedding", 
            index_params={
                "index_type": "IVF_FLAT",  # Index type: Inverted File Flat
                "metric_type": "L2",  # Distance metric: L2 (Euclidean)
                "params": {"nlist": 1024}  # Number of clusters in the index
            }
        )

        # Verify the index creation
        check_index(collection_name, "abstract_embedding")
        
        return collection

    except Exception as e:
        print(f"Error in collection creation/loading: {e}")
        raise

def prepare_insertion_data(csv_file):
    """
    Prepares data from a CSV file for insertion into the Milvus collection.

    This function:
    1. Reads the CSV file using pandas.
    2. Processes each row to extract relevant fields.
    3. Chunks long abstracts into smaller parts.
    4. Generates embeddings for each chunk using the embedding model.
    5. Organizes data into a format suitable for insertion into Milvus.

    Args:
        csv_file (str): Path to the CSV file containing research paper data.

    Returns:
        list: A list of lists where each inner list contains values for a specific field.
    """
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Lists to store processed data
    paper_ids = []
    titles = []
    abstracts = []
    embeddings = []
    years = []
    links = []

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        # Chunk the abstract into smaller parts
        chunks = chunk_message(row["abstract"])
        
        for chunk in chunks:
            # Generate embedding for the chunk using the NLP model
            chunk_embedding = embedding_model.process_text(chunk)

            # Append data to respective lists
            paper_ids.append(row["paperID"])
            titles.append(row["title"])
            abstracts.append(chunk)  # Store the chunk instead of the full abstract
            embeddings.append(chunk_embedding)
            years.append(int(row["year"]))
            links.append(row["link"])

    # Prepare the final data structure for insertion
    entities = [paper_ids, titles, abstracts, embeddings, years, links]

    return entities

def insert_into_collection(collection_name, batch_size=100):
    """
    Inserts data into a Milvus collection in batches for efficiency.

    This function:
    1. Loads or creates the specified Milvus collection.
    2. Reads and processes data from a CSV file.
    3. Inserts data into the collection in batches to optimize performance.
    4. Flushes the collection to ensure data persistence.

    Args:
        collection_name (str): The name of the Milvus collection.
        batch_size (int, optional): The number of records to insert per batch. Default is 100.

    Returns:
        bool: True if insertion is successful, False otherwise.
    """
    # Load or create the collection
    collection = create_or_load_collection(collection_name)
    
    # Prepare data for insertion
    data = prepare_insertion_data(csv_file="outputs.csv")

    try:
        total_records = len(data[0])  # Number of records to insert

        # Insert data in batches
        for i in range(0, total_records, batch_size):
            batch_data = [field[i:i+batch_size] for field in data]
            collection.insert(batch_data)

        # Ensure data is written to disk
        collection.flush()
        return True

    except Exception as e:
        print(f"An error occurred during insertion: {e}")
        return False


async def search_similar_texts(query_text, top_k=50):
    """
    Searches for similar texts in the Milvus collection based on embeddings.

    This function:
    1. Converts the input query text into an embedding using an NLP model.
    2. Calls the `search_with_similarity` function to find the top_k most similar embeddings.

    Args:
        query_text (str): The input text to search for similar entries.
        top_k (int, optional): The number of top similar results to return. Default is 50.

    Returns:
        list: A list of search results containing relevant information from the collection.
    """
    # Convert query text to embedding
    query_embedding = embedding_model.process_text(query_text)

    # Perform similarity search
    result2 = await search_with_similarity(query_embedding, top_k)
    return result2


async def search_with_similarity(query_embedding, top_k):
    """
    Performs a similarity search in the Milvus collection.

    This function:
    1. Connects to the Milvus server.
    2. Loads the "HealthTech" collection into memory.
    3. Searches for the top_k most similar embeddings using the L2 metric.
    4. Returns results containing the title, abstract, year, and link.

    Args:
        query_embedding (list): The embedding vector representation of the query text.
        top_k (int): The number of top results to return.

    Returns:
        list or None: A list of search results or None if an error occurs.
    """
    try:
        # Establish connection to the Milvus server
        connections.connect(host="localhost", port=29530)
        
        # Load the collection into memory
        collection = Collection(name="HealthTech")
        collection.load()

        # Perform similarity search based on embeddings
        query_results = collection.search(
            data=[query_embedding],  # Input query vector
            anns_field="abstract_embedding",  # Field to search in
            param={"metric_type": "L2", "params": {"nprobe": 50}},  # Search parameters
            limit=top_k,  # Number of results to return
            output_fields=["title", "abstract", "year", "link"]  # Fields to return in results
        )

        # Process and return results if found
        if query_results:
            return await process_results(query_results)

        return None  # No results found
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return None

async def process_results(query_results, output_file="output.csv"):
    """
    Processes search results, groups entries by title, year, and link,
    and writes the output to a CSV file without including paperID.

    This function:
    1. Groups search results based on (title, year, link).
    2. Aggregates abstracts for each unique entry.
    3. Saves the processed results into a CSV file.

    Args:
        query_results (list): List of search results obtained from Milvus.
        output_file (str, optional): Path to the CSV file where results will be stored. Default is "output.csv".

    Returns:
        list: A list of grouped results with title, year, link, and combined abstracts.
    """
    # Dictionary to store grouped results
    grouped_results = defaultdict(lambda: {"title": None, "year": None, "link": None, "abstracts": []})

    for result in query_results:
        for match in result:
            title = match.entity.get("title")
            year = match.entity.get("year")
            link = match.entity.get("link")
            abstract = match.entity.get("abstract")

            key = (title, year, link)  # Unique key to group results

            # Initialize entry only once
            if grouped_results[key]["title"] is None:
                grouped_results[key].update({"title": title, "year": year, "link": link})

            # Append abstracts to the list
            if abstract:
                grouped_results[key]["abstracts"].append(abstract)

    # Convert grouped results into a list
    results_list = list(grouped_results.values())

    # Write processed results to a CSV file
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write header (excluding paperID)
        writer.writerow(["title", "year", "link", "abstracts"])
        
        # Write rows with JSON-encoded abstracts
        for entry in results_list:
            writer.writerow([
                entry["title"], 
                entry["year"], 
                entry["link"], 
                json.dumps(entry["abstracts"])  # Store abstracts as JSON
            ])
    
    return results_list


def check_index(collection_name, field_name):
    """
    Checks whether an index exists on a specified field in a Milvus collection.

    Args:
        collection_name (str): The name of the collection to check.
        field_name (str): The name of the field to verify.

    Returns:
        bool: True if an index exists on the given field, False otherwise.
    """
    collection = Collection(collection_name)
    indexes = collection.indexes

    # Iterate over all indexes and check if the field name matches
    for index in indexes:
        if index.field_name == field_name:
            return True
    return False


def chunk_message(abstract_text, chunk_size=250):
    """
    Splits an abstract into smaller text chunks of a specified size.

    This function:
    1. Splits the text into words.
    2. Iteratively adds words to a chunk until it reaches the maximum allowed size.
    3. Stores each chunk separately.

    Args:
        abstract_text (str): The full abstract text to be chunked.
        chunk_size (int, optional): The maximum number of characters per chunk. Default is 250.

    Returns:
        list: A list of text chunks.
    """
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

    # Append the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def drop_collection(collection_name):
    """
    Drops (deletes) a Milvus collection.

    This function:
    1. Connects to the Milvus server.
    2. Attempts to drop the specified collection.

    Args:
        collection_name (str): The name of the collection to be deleted.

    Returns:
        None
    """
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
    """
    Performs a comprehensive diagnostic check on a Milvus collection.

    This function:
    1. Connects to the Milvus server.
    2. Loads the specified collection and ensures it is ready for operations.
    3. Retrieves and prints collection statistics, including the total number of entities.
    4. Runs a sample query to verify the integrity and accessibility of the data.

    Args:
        collection_name (str): The name of the collection to diagnose.

    Returns:
        None
    """
    try:
        # Ensure connection to the Milvus server
        connect_to_server()
        
        # Get the collection object
        collection = Collection(name=collection_name)
        
        # Load collection into memory
        collection.load()
        collection.flush()  # Ensure all pending operations are committed
        
        # Display collection statistics
        print("\n--- Collection Diagnostics ---")
        print(f"Collection Name: {collection_name}")
        print(f"Total Entities: {collection.num_entities}")  # Number of records in the collection
    
        try:
            # Perform a sample query to check if data retrieval works correctly
            query_result = collection.query(
                expr="id >= 0",  # Query condition (adjust as needed)
                output_fields=["id", "paperID", "title", "abstract", "abstract_embedding", "year"], 
                limit=10  # Limit results to avoid excessive output
            )

            # Print retrieved entities (truncating embeddings for readability)
            for entity in query_result:
                print(f"""
                paperID: {entity['paperID']}
                Title: {entity['title']}
                Abstract text: {entity['abstract']}
                Embedding: {entity['abstract_embedding'][:5]}... (truncated)
                Year: {entity['year']}
                """)

        except Exception as query_error:
            print(f"Query Error: {query_error}")

    except Exception as e:
        print(f"Diagnostic Error: {e}")