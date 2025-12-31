import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Starting model check...")
try:
    models = list(genai.list_models())
    print(f"Found {len(models)} models.")
    for m in models:
        print(f"Model: {m.name} | Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
