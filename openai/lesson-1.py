from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# input messages to OpenAI here
input_messages = [
    {
        "role": "user",
        "content": "how much wood would a woodchuck chuck "
    }
]

#add properties of OpenAI here
response =  client.responses.create(
    model = "o3-mini",
    # instructions="Respond like a very gentle and romantic guy", #affect behaviours and rules of the model
    input = input_messages,
    reasoning = {
        'effort': 'low'
    }
    # stream = True #Stream the response back to us chunk by chunk
)

print(response.output_text)

# With Streaming
# full_response = ""
#
# for event in response:
#     if event.type == "response.output_text.delta":
#         print(event.delta, end="", flush=True)
#         full_response += event.delta
#
# print("\n\nFull Response: ", full_response)