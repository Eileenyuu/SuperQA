import os
import requests
import csv
from apikeys import S2_API_KEY

csv_file = "outputsss.csv"
S2_API_KEY = os.environ['S2_API_KEY'] = S2_API_KEY
result_limit = 100  # each page(max 100)
required_count = 50  # complete ref num
def preprocess_text(text, field_name):
    if not text or str(text).lower() == "none":
        return f"{field_name} not available"
    text = str(text).strip().replace('\n', ' ').replace('\r', '')
    return ' '.join(text.split())

def main():
    query = input("Find papers about what: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return
    get_paper_information_paginated(query)

def get_paper_information_paginated(query):
    all_data = []
    offset = 0

    print(f"Searching for at least {required_count} complete papers...")

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
            print(f"Collected {len(all_data)}/{required_count}: {filtered_data['title']}")

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
        print(f"\nCSV file '{csv_file}' created with {len(all_data)} complete records.")
    else:
        print("No complete records collected.")

if __name__ == '__main__':
    main()
