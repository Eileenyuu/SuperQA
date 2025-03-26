from milvus_database import *
from preprocess_text import *
from openai_models import extract_biomedical_terms, mcq_openai_output
from anthropic_models import mcq_anthropic_output
from helper_functions import csv_to_string
import asyncio  # Required for running async functions

async def openai_lab_bench_testing(query): 
    final_query = query.question + '\n' + '\n'.join(query.choices)   
    key_terms = extract_biomedical_terms(query.question)
    get_paper_information_paginated(key_terms, result_limit=50, required_count=20)

    connect_to_server()
    insert_into_collection("HealthTech")
    await search_similar_texts(key_terms)

    formatted_string = csv_to_string()
    drop_collection("HealthTech")

    return mcq_openai_output(user_query=final_query, final_context=formatted_string)

async def anthropic_lab_bench_testing(query): 
    final_query = query.question + '\n' + '\n'.join(query.choices)   
    key_terms = extract_biomedical_terms(query.question)
    get_paper_information_paginated(key_terms, result_limit=50, required_count=20)

    connect_to_server()
    insert_into_collection("HealthTech")
    await search_similar_texts(key_terms)

    formatted_string = csv_to_string()
    drop_collection("HealthTech")

    return mcq_anthropic_output(user_query=final_query, final_context=formatted_string)

async def evaluate_mcq_performance(csv_file):
    correct_count = 0
    total_count = 0

    # Open and read the CSV file
    with open(csv_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the header

        for row in reader:
            print("")
            question = row[0]
            choices = row[1].split("\n")  # Choices are stored with newline separation
            correct_answer = row[2]  # The correct answer (A, B, C, or D)

            # Create a query-like object
            class Query:
                def __init__(self, question, choices):
                    self.question = question
                    self.choices = choices

            query = Query(question, choices)

            # Run the MCQ test
            model_answer = await anthropic_lab_bench_testing(query)

            print(model_answer)
            print(correct_answer)

            # Extract the answer from the model output (assuming it returns A, B, C, D)
            predicted_answer = model_answer.strip()

            # Compare the predicted answer to the correct answer
            if predicted_answer == correct_answer:
                correct_count += 1

            total_count += 1

    # Calculate and print accuracy
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"MCQ Evaluation Accuracy: {accuracy:.2f}%")

# Run evaluation
csv_file = "questions.csv"
asyncio.run(evaluate_mcq_performance(csv_file))