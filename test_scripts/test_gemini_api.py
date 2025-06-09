import google.generativeai as genai

# Configure the API key
genai.configure(api_key='AIzaSyBHSYxSb9lQ7-QN_nqp9OoEZVlF2i6BQao')

# Create a model instance
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Test the API with a simple prompt
response = model.generate_content('Hello, how are you?')

# Print the response
print(response.text) 