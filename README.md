# SuperQA

![alt text](https://github.com/Eileenyuu/SuperQA/blob/main/hackathon.jpg)

This is a HealthTech AI Hub Hackathon Challenge.  
Hackathon Challenge 3.1 : Create a MVP for a AI-Powered Literature Review Assistant.


> [!WARNING]
> API Keys will not be provided in the repository and should be put in by the user.
> Input your API Key into the src/apikeys.py file.

Running the Program:

## Step 1:
Clone the repository
```
git clone git@github.com:Eileenyuu/SuperQA.git
```

## Step 2:
Set up the docker network
First downlaod docker desktop    
Once docker desktop or docker is downloaded, in your terminal 
```
cd SUPERQA
docker network create milvus  
docker-compose up -d  
docker-compose -f docker-compose.yml up  
```
  
## Step 3: 
Setup the environment 
> [!NOTE] 
> If you are using VScode, make sure you have navigated to any python file in the repository and open it before activating the environment and installing the dependencies in the requirement.txt file.  
> Else you would get this error: Defaulting to user installation because normal site-packages is not writeable
``` 
python -m venv venv  
```
Make sure to kill the terminal and start a new one to initialise the venv environment     
```
source venv/bin/activate 
```

## Step 4:  
Run this command to install all the libraries required for this project.
```  
pip install -r requirements.txt 
``` 

> [!NOTE] 
> To run the project, make sure the Milvus database is up and running,  
> You can check docker desktop for a more readable way to check the container is running  
  
> Then go to main.py if you wish to run the project from the terminal.  
  
> If you want to run the frontend/Flask application, go to app.py and run the file  
> Wait for the terminal to show that the Flask application has been set up  
> Open 127.0.0.1/5000 to access the frontend  

> Finally, to run the Lab Bench, go into create_litqa2_dataset.py and run the file  
> If it works, a file called questions.csv will be generated  
> Then, go to lab_bench_test.py and run the code to identify how accurate the AI agent is  
> Spoiler alert, it probably won't be that accurate :skull:  

## Results: 
OpenAI GPT 4o Results:  
39.20% accuracy  
Anthropic Claude 3.5 Haiku Results:  
23.62% accuracy  

