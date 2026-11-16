
from google import genai
from dotenv import load_dotenv
import os

from database import askQuestion
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=API_KEY)


def create_prompt(question: str) -> str:

    print("Searching documents...")

    chunks = askQuestion(question)

    print("Retrieved chunks:", len(chunks))

    context = ""

    for doc in chunks:
        context += doc.page_content + "\n\n"

    if not context.strip():
        context = "No relevant information was found."

    prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant.

        Answer the user's question using only the
        information provided in the context.

        If the answer is not available in the context,
        respond with:
        "I don't know based on the provided information."

        Context:
        {context}

        Question:
        {question}

        Answer:
    """)

    formatted_prompt = prompt_template.invoke({
        "context": context,
        "question": question
    })

    return formatted_prompt.to_string()


def findResponse(question: str) -> str:

    print("Creating prompt...")

    final_prompt = create_prompt(question)

    print("Sending request to Gemini...")

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=final_prompt
    )

    print("Gemini response received!")

    return response.text or "No response generated."
