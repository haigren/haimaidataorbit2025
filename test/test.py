import os

from groq import Groq

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "can you write this in a friendlier manner?"
        }
    ],
    model="llama-3.3-70b-versatile",
)

# print(len(chat_completion.choices))
print(chat_completion.choices[0].message.content)