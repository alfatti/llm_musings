from google import genai
from google.auth.credentials import Credentials
from google.genai.types import GenerateContentConfig

# Step 1: Vault-issued access token
access_token = "YOUR_BEARER_TOKEN_FROM_VAULT"

# Step 2: Create a credentials object from the token
creds = Credentials(token=access_token)

# Step 3: Configure endpoint and client
client = genai.Client(
    credentials=creds,
    client_options={
        "api_endpoint": "us-central1-aiplatform.googleapis.com"  # change if yours is different
    },
    http_options={"api_version": "v1"}
)

# Step 4: Call your specific Gemini model
model_path = "projects/YOUR_PROJECT_ID/locations/us-central1/publishers/google/models/gemini-1.5-flash"

response = client.models.generate_content(
    model=model_path,
    contents=[
        {
            "role": "user",
            "parts": [{"text": "Tell me a fun fact about California real estate."}]
        }
    ],
    config=GenerateContentConfig(
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        max_output_tokens=1024,
        stop_sequences=[]
    )
)

# Step 5: Print the response
print(response.candidates[0].content.parts[0].text)
