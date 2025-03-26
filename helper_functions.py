import csv
import re

def csv_to_string(output_file="output.csv"):
    title_year_abstracts = ""
    with open(output_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if the CSV file has one
        
        for row in reader:
            title, year, link, abstracts = row  # Assuming CSV has 4 columns in order
            # Append data to respective strings
            title_year_abstracts += f"{title},{year},{abstracts}\n"

    return title_year_abstracts

def add_br_before_key_terms(suggestion):
    # List of key terms to match
    key_terms = ["Title:", "Explanation:", "Some key points related to the topic:", 
                 "Description:", "Insights:", "Source:"]
    
    # Loop over each key term and add <br> before and after them
    for term in key_terms:
        suggestion = re.sub(f"(?={re.escape(term)})", "<br><br>", suggestion)  # Add before
        suggestion = re.sub(f"(?<=({re.escape(term)}))", "<br><br>", suggestion)  # Add after
    
    # Add <br> before and after numbered points (1., 2., 3., etc.)
    suggestion = re.sub(r"(?=\d\.)", "<br><br>", suggestion)  # Add before numbered points
    
    return suggestion