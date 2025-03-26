from openai import OpenAI
from apikeys import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

prompt_template = """
You are an expert in biomedical terminology and information retrieval. Your task is to extract the most relevant key terms from a user query to improve search performance in a vector database.

## **Instructions:**
- Identify the most important biomedical terms, including diseases, symptoms, treatments, drugs, biological processes, genes, and medical conditions.
- Exclude generic words, stopwords, and non-informative terms.
- Return the extracted terms as a strings.
- Do not include any explanation, only output the list.

## **User Query:**
"{query}"

## **Output Format:**
"term1, term2, term3, term4, term5, term6"
"""

prompt_template2 = """
You are an expert in biomedical sciences, specializing in question answering based on research literature. Your task is to analyze the provided abstracts and accurately answer the user's query using relevant biomedical information.

Instructions:
- Use the provided abstracts to generate a clear, concise, and well-structured response.
- Format your response as follows:

Title:
Full Title here

Explanation:
Full Explanation here

Some key points related to the topic:

1. Key_Point_1_Title
    Description: Key_Point_1_Description
    Insights: Relevant_Insight_or_Impact_1
    Source: [Title_1, Title_1_Year], [Title_2, Title_2_Year]

2. Key_Point_2_Title
    Description: Key_Point_2_Description
    Insights: Relevant_Insight_or_Impact_
    Source: [Title_1, Title_1_Year], [Title_2, Title_2_Year]
   
- If the answer is not directly available in the abstracts, state that explicitly instead of making assumptions.  
- Ensure that complex biomedical terms are explained in a way that is understandable to a professional or a well-informed user.  
- Maintain a formal and authoritative tone in your response.  

## **Context:**  
{context}  

## **User Query:**  
"{query}"  

## **Your Answer:**  
"""

def text_to_embedding(text):
  response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=text,
    encoding_format="float"
  )

  return(response.data[0].embedding)

def extract_biomedical_terms(query):
    prompt = prompt_template.format(query=query)
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return completion.choices[0].message.content

def final_openai_output(user_query, final_context):
    prompt = prompt_template2.format(query=user_query, context=final_context)
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return completion.choices[0].message.content