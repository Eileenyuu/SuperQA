from openai import OpenAI
import base64

from dotenv import load_dotenv
load_dotenv()

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("./nikki.jpg")

client = OpenAI()

input_messages = [
    {
        'role': 'user',
        'content': [
            {
                "type": "input_text",
                "text": "Please describe these images separately."
            },

            #passing image as filepath
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            #passing image as URL
            {
                "type": "input_image",
                "image_url": "https://belvoir-university-health.s3.amazonaws.com/media/2020/07/08144529/lemon-health-benefits-1.jpg"
            }
        ]
    }
]

response = client.responses.create(
    model="gpt-4o-mini",
    input=input_messages
)

print(response.output_text)