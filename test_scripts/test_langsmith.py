import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "legal-research-agent"
os.environ["GEMINI_API_KEY"] = "AIzaSyDRP9UsqU4j9MAvxgFRecPCqGlWZzLZdYE"

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=os.environ["GEMINI_API_KEY"])

print("Starting the script...")
print("Environment variables set.")
print("Invoking the model...")
response = llm.invoke("Hello, world!")
print("Response received:", response)