import asyncio
from flask import Flask, request, jsonify, render_template
from helper_functions import csv_to_string, add_br_before_key_terms
from preprocess_text import get_paper_information_paginated
from openai_models import extract_biomedical_terms, final_openai_output
from milvus_database import connect_to_server, insert_into_collection, search_similar_texts, drop_collection

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration parameters
REQUIRED_COUNT = 30
RESULT_LIMIT = 100

@app.route("/")
def home():
    # Render index.html from the "templates" folder
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query."}), 400

    try:
        # 1. Extract key biomedical terms using OpenAI
        key_terms = extract_biomedical_terms(query)
        
        # # 2. Retrieve paper information from Semantic Scholar API and write to outputs.csv
        get_paper_information_paginated(key_terms, required_count=REQUIRED_COUNT, result_limit=RESULT_LIMIT)
        
        # 3. Connect to Milvus and insert the retrieved data into the "HealthTech" collection
        connect_to_server()
        insert_into_collection("HealthTech")
        
        # 4. Execute vector similarity search (asynchronous)
        search_results = asyncio.run(search_similar_texts(key_terms))
        
        # 5. Read the CSV file as context and generate a suggested answer via OpenAI
        abstracts = csv_to_string()

        # Generate response from OpenAI using abstracts
        suggestion = final_openai_output(user_query=query, final_context=abstracts)
        final_suggestion = add_br_before_key_terms(suggestion)
        
        # 6. Process search results by merging abstract chunks into a single string
        papers = []
        if search_results:
            for paper in search_results:
                abstracts = paper.get("abstracts", [])
                if isinstance(abstracts, list):
                    paper["abstract"] = " ".join(abstracts)
                else:
                    paper["abstract"] = abstracts
                if "abstracts" in paper:
                    del paper["abstracts"]
                papers.append(paper)
        
        # 7. Clean up by dropping the Milvus collection to avoid duplication
        drop_collection("HealthTech")
        
        # Return the results and suggestion in JSON format
        return jsonify({"papers": papers, "suggestion": final_suggestion})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
