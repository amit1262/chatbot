from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    """
    Define the state of the chatbot.
    This can be extended with more fields as needed.
    """

    messages: Annotated[list, add_messages]


def load_dotenv(override: bool = True):
    # Load environment variables from .env file
    status = load_dotenv(override=True)
    assert status, "Failed to load environment variables from .env file"


def _get_system_prompts():
    # Define a system message to set the context for the conversation
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a professional assistant with deep expertise in {domain}."
    )
    # decide a domain for the chatbot
    chatbot_domain = input(
        "Enter the domain for the chatbot (e.g., medical, sports, science): "
    ).strip()
    system_message = system_prompt.format(domain=chatbot_domain)
    assert system_message, "Failed to create system message"
    return system_message


def get_initial_state() -> State:
    """
    Initialize the state of the chatbot.
    This can be extended to include more fields as needed.
    """
    system_prompts = _get_system_prompts()
    return {"messages": [system_prompts]}


def invoke_model(current_state: State) -> State:
    """
    Invoke the model with the current state.
    This function can be extended to include more complex logic.
    """
    messages = current_state.get("messages", [])
    if not messages:
        raise ValueError("No messages found in the state.")
    print(f"Agent: ")
    full_response = ""
    for chunk in model.stream(messages):
        token = chunk.text()  # or chunk.text() in some versions
        print(token, end="", flush=True)
        full_response += token

    return {"messages": [AIMessage(content=full_response)]}


def get_user_input(current_state: State) -> State:
    """
    Get user input and update the state.
    This function can be extended to include more complex logic.
    """
    user_input = input("\n\nUser: ").strip()
    if not user_input:
        raise ValueError("User input cannot be empty.")

    # Append the user message to the current state
    return {"messages": [HumanMessage(content=user_input)]}


def initialize_model():
    """
    Initialize the model using environment variables.
    This function can be extended to include more complex logic.
    """
    # model initialization
    model = ChatOllama(
        model="llama3.2:latest", temperature=0.1, max_tokens=1000, streaming=True
    )
    assert model, "Failed to initialize ChatOllama model"
    print("ChatOllama model initialized successfully.")
    return model


if __name__ == "__main__":
    load_dotenv
    model = initialize_model()

    # create a workflow graph
    workflow = StateGraph(state_schema=State)
    workflow.add_node("model", invoke_model)
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)

    # invoke the graph
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "bot_1"}}
    workflow_compiled = workflow.compile(checkpointer=memory)
    first_run = config.get("configurable", {}).get("thread_id") not in memory.storage
    initial_state = get_initial_state()

    while True:
        try:
            user_input = input("\n\nUser: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if first_run:
                initial_state["messages"].append(HumanMessage(content=user_input))
                state = initial_state
                print("First run, state:", state)
            else:
                state = {"messages": [HumanMessage(content=user_input)]}
                print("state:", state)
            workflow_compiled.invoke(state, config=config)
        except:
            print("An error occurred. Please try again.")
            break
