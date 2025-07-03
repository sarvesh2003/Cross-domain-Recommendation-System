# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load .env and API key
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# # Configure the API client
# genai.configure(api_key=api_key)

# # List available models
# models = genai.list_models()

# for model in models:
#     print(f"Model: {model.name}")
#     print(f"  Description: {model.description}")
#     print(f"  Capabilities: {model.supported_generation_methods}")
#     print("-" * 40)





# import os
# from dotenv import load_dotenv
# # Get API key
# load_dotenv()  # Loads variables from .env into environment
# apiKey = os.environ.get("OPENAI_API_KEY")


# import openai
# from openai import OpenAI

# client = OpenAI(api_key=apiKey) 

# models = client.models.list()
# for model in models.data:
#     print(model.id)

