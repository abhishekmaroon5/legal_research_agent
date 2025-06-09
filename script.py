from groq import Groq
import ssl
 
# Create custom SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
 
client = Groq(api_key="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")
completion = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
      {
        "role": "user",
        "content": "Hi"
      }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    stream=True,
    stop=None,
)
 
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")