import os
import requests
import csv
from apikeys import S2_API_KEY

csv_file = "outputs.csv"
S2_API_KEY = os.environ['S2_API_KEY'] = S2_API_KEY

def preprocess_text(text, field_name):
    """
    Cleans and formats a given text field. If the text is empty or "none", 
    it replaces it with a default message.

    Args:
        text (str): The text content to preprocess.
        field_name (str): The name of the field for error messages.

    Returns:
        str: The cleaned and formatted text, or a placeholder message if missing.
    """
    if not text or str(text).lower() == "none":
        return f"{field_name} not available"
    
    # Remove leading/trailing whitespace and normalize line breaks
    text = str(text).strip().replace('\n', ' ').replace('\r', '')
    
    # Remove excessive spaces
    return ' '.join(text.split())


def get_paper_information_paginated(query, required_count, result_limit):
    """
    Retrieves a list of research papers from the Semantic Scholar API using pagination.
    Filters out incomplete records and saves the results in a CSV file.

    Args:
        query (str): The search term for finding research papers.
        required_count (int): The total number of papers to retrieve.
        result_limit (int): The number of papers to request per API call.

    Returns:
        None: Writes the collected paper data to a CSV file.
    """
    all_data = []  # Stores all valid paper entries
    offset = 0  # Tracks the pagination offset

    while len(all_data) < required_count:
        # Make API request to Semantic Scholar
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={'X-API-KEY': S2_API_KEY},
            params={
                'query': query,
                'limit': result_limit,
                'offset': offset,
                'fields': 'paperId,title,year,abstract,openAccessPdf'
            }
        )

        # Handle request errors
        if rsp.status_code != 200:
            print(f"Request failed at offset {offset}: {rsp.status_code} - {rsp.text}")
            break

        results = rsp.json()
        papers = results.get("data", [])  # Extract paper data
        total = results.get("total", 0)   # Total available papers

        # Stop if there are no more papers to process
        if not papers:
            print("No more papers available.")
            break

        for paper in papers:
            # Ensure that the paper has all required fields
            if not all([
                paper.get("paperId"),
                paper.get("title"),
                paper.get("year"),
                paper.get("abstract"),
                paper.get("openAccessPdf", {}).get("url")
            ]):
                print(f"Skipping incomplete paper: {paper.get('paperId')}")
                continue  # Skip incomplete records

            # Process and clean extracted data
            filtered_data = {
                "paperID": preprocess_text(paper.get("paperId"), "Paper ID"),
                "title": preprocess_text(paper.get("title"), "Title"),
                "year": preprocess_text(paper.get("year"), "Year"),
                "abstract": preprocess_text(paper.get("abstract"), "Abstract"),
                "link": preprocess_text(paper.get("openAccessPdf", {}).get("url"), "Link"),
            }

            all_data.append(filtered_data)

            # Stop once we've collected enough papers
            if len(all_data) >= required_count:
                break

        # Update offset for pagination
        offset += result_limit

        # Stop if we've reached the total available results
        if offset >= total:
            print("Reached end of available results.")
            break

    # Save results to a CSV file if data is available
    if all_data:
        fieldnames = ["paperID", "title", "year", "abstract", "link"]
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"Successfully saved {len(all_data)} papers to {csv_file}")
    else:
        print("No complete records collected.")