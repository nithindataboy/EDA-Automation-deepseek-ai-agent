import requests

API_KEY = "sk-bXb1pHpASDFAW87M8rDFZASDFbid9ASDFG4T3BlbkFJCUATRSDFKAJASDFOR40CCCruyiOPYA"  # Replace with your actual API key
API_URL = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-3.5-turbo",  # You can also use "gpt-4"
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
}

response = requests.post(API_URL, headers=headers, json=data)

if response.status_code == 200:
    print("✅ API Key is working!")
    print("Response:", response.json())
else:
    print(f"❌ API Key is NOT working! Status Code: {response.status_code}")
    print("Error:", response.text)
