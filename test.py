import json
import csv
import string
import random

# Define input and output file paths
jsonl_file = "litqa-v2-public.jsonl"
csv_file = "questions.csv"

# Open the JSONL file and process each line
with open(jsonl_file, "r", encoding="utf-8") as infile, open(csv_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    
    # Write the CSV header
    writer.writerow(["question", "choices", "answer"])
    
    for line in infile:
        data = json.loads(line.strip())
        
        # Extract question
        question = data.get("question", "")

        answer = data.get("ideal", "")
        
        # Combine ideal and distractors as multiple-choice options
        choices = [answer] + data.get("distractors", [])

        random.shuffle(choices)

        formatted_choices = [f"{letter}. {choice}" for letter, choice in zip(string.ascii_uppercase, choices)]

        answer_letter = string.ascii_uppercase[choices.index(answer)]
        
        # Convert choices into a comma-separated string
        choices_str = ", ".join(formatted_choices)
        
        # Write to CSV
        writer.writerow([question, choices_str, answer_letter])

print(f"CSV file saved: {csv_file}")