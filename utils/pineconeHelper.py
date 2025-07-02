from dotenv import load_dotenv
import os

class PineConeOperations:
    def __init__(self):
        load_dotenv()  # Loads variables from .env into environment
        self.api_key = os.environ.get("PINECONE_API_KEY")
    
    def fetch_based_on_index(self, index_name, ID)
