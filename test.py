import requests

url = "http://127.0.0.1:8000/generate"
data = {"text": "What is a cat?", "max_length": 20}

response = requests.post(url, json=data)
generated_text = response.json()["generated_text"]
print(generated_text)
