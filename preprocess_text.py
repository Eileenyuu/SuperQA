import os
import requests
import csv
from apikeys import S2_API_KEY

csv_file = "outputs.csv"
S2_API_KEY = os.environ['S2_API_KEY'] = S2_API_KEY

def preprocess_text(text, field_name):
    if not text or str(text).lower() == "none":
        return f"{field_name} not available"
    text = str(text).strip().replace('\n', ' ').replace('\r', '')
    return ' '.join(text.split())

def get_paper_information_paginated(query, required_count, result_limit):
    all_data = []
    offset = 0

    while len(all_data) < required_count:
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

        if rsp.status_code != 200:
            print(f"Request failed at offset {offset}: {rsp.status_code} - {rsp.text}")
            break

        results = rsp.json()
        papers = results.get("data", [])
        total = results.get("total", 0)

        if not papers:
            print("No more papers available.")
            break

        for paper in papers:
            if not all([
                paper.get("paperId"),
                paper.get("title"),
                paper.get("year"),
                paper.get("abstract"),
                paper.get("openAccessPdf", {}).get("url")
            ]):
                print(f"Skipping incomplete paper: {paper.get('paperId')}")
                continue

            filtered_data = {
                "paperID": preprocess_text(paper.get("paperId"), "Paper ID"),
                "title": preprocess_text(paper.get("title"), "Title"),
                "year": preprocess_text(paper.get("year"), "Year"),
                "abstract": preprocess_text(paper.get("abstract"), "Abstract"),
                "link": preprocess_text(paper.get("openAccessPdf", {}).get("url"), "Link"),
            }

            all_data.append(filtered_data)

            if len(all_data) >= required_count:
                break

        offset += result_limit
        if offset >= total:
            print("Reached end of available results.")
            break

    if all_data:
        fieldnames = ["paperID", "title", "year", "abstract", "link"]
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(all_data)
    else:
        print("No complete records collected.")