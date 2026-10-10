import os

try:  # optional - lets you keep secrets in a .env file
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("your api key")
BASE_URL = os.getenv("base url")
MODEL = os.getenv("model")
