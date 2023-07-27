import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERPAPI_API_KEY")