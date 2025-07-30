from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)

if __name__ == "__main__":
    # Load environment variables from .env file
    status = load_dotenv(override=True)
    assert status, "Failed to load environment variables from .env file"

    # model initialization
    model = ChatOllama(
        model="llama3.2:latest", temperature=0.1, max_tokens=1000, streaming=True
    )
    assert model, "Failed to initialize ChatOllama model"
    print("ChatOllama model initialized successfully.")

    # decide a domain for the chatbot
    chatbot_domain = input(
        "Enter the domain for the chatbot (e.g., medical, sports, science): "
    ).strip()

    # Define a system message to set the context for the conversation
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a professional assistant with deep expertise in {domain}."
    )
    system_message = system_prompt.format(domain=chatbot_domain)
    assert system_message, "Failed to create system message"
    messages = [system_message]

    while True:
        user_message = input("\n\nUser: ").strip()
        user_message = HumanMessage(content=user_message)
        messages.append(user_message)

        print(f"Agent: ")
        full_response = ""
        for chunk in model.stream(messages):
            token = chunk.text()  # or chunk.text() in some versions
            print(token, end="", flush=True)
            full_response += token

        messages.append(AIMessage(content=full_response))
