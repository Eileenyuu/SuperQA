# SuperQA
This is a HealthHub Hackathon Project  
API Keys will not be provided in the repository and should be put in by the user  
  
Project Goal is to develop a MVP to display the strength of RAG with LLMs in a BioMed environment  
  
Step 1 to getting the project to work:    
First downlaod docker desktop    
Once docker desktop or docker is downloaded    
  
Step 2 (Run these commands in terminal):  
docker network create milvus  
docker-compose up -d  
    
Run the docker for the milvus database:    
docker-compose -f docker-compose.yml up  
  
Step 3:  
Setup a venv environment in VScode  
python -m venv venv  
Make sure to kill the terminal and start a new one to initialise the venv environment     
Linux: source venv/bin/activate  

Step 4:  
Run this command to install all the libraries required for this project  
Windows: pip install -r requirements.txt  

Things to note when running the project:  
To run the project, make sure the Milvus database is up and running,  
You can check docker desktop for a more readable way to check the container is running  
  
Then go to main.py if you wish to run the project from the terminal.  
  
If you want to run the frontend/Flask application, go to app.py and run the file  
Wait for the terminal to show that the Flask application has been set up  
Open 127.0.0.1/5000 to access the frontend  

Finally, to run the Lab Bench, go into create_litqa2_dataset.py and run the file  
If it works, a file called questions.csv will be generated  
Then, go to lab_bench_test.py and run the code to identify how accurate the AI agent is  
Spoiler alert, it probably won't be that accurate :skull:  
  
OpenAI GPT 4o Results:  
39.20% accuracy  
Anthropic Claude 3.5 Haiku Results:  
No clue, semanticscholar broke again last night :skull:  

P.S.  
If you are running python -m venv venv in VScode, make sure you have navigated to any python file and opened it  
Then open a new terminal and run the command python -m venv venv  
Then you can run pip install -r requirements.txt  
I have no idea why but it works only if you do that.  
Else you would get this error: Defaulting to user installation because normal site-packages is not writeable  
