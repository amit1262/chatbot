from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    """
    Define the state of the chatbot.
    This can be extended with more fields as needed.
    """

    messages: Annotated[list, add_messages]


def get_initial_state() -> State:
    """
    Initialize the state of the chatbot.
    This can be extended to include more fields as needed.
    """
    # Define a system message to set the context for the conversation
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an experienced clinical researcher with lots of knowledge for conducting network studies, in OHDSI ecosystem, using R and Strategus packages."
        "Today, you will be helping a user with generating a complete and executable R code for a Strategus network study."
        "let's start by asking the user about what kind of network study they want to conduct."
        "Given the user's input, first estimate if question is complete, if not, ask the user for more details."
        "Your job is to guide the user step by step - for each step, ask the user for specific details related to that step and then generate the R code based on their input."
        "At any point, just discuss only the next step and wait for the user to provide the necessary details."
        "No need to explain the conceptual details about any step - unless asked by the user."
        "Assume the user doesn't know anything about Strategus or R and guide them through the process - one step at a time."
    )
    return {"messages": [system_prompt.format()]}


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
    print("\n")
    print("*" * 100)
    print("\n")

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
    MODEL_NAME = "azure_openai:gpt-4o"
    model = init_chat_model(model=MODEL_NAME)
    assert model, "Failed to initialize ChatOllama model"
    print(f"{MODEL_NAME} initialized successfully.")
    return model


if __name__ == "__main__":
    load_dotenv()
    model = initialize_model()
    print("*" * 100)

    # create a workflow graph
    workflow = StateGraph(state_schema=State)
    workflow.add_node("model", invoke_model)
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)

    # invoke the graph
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "bot_2"}}
    workflow_compiled = workflow.compile(checkpointer=memory)
    first_run = config.get("configurable", {}).get("thread_id") not in memory.storage
    initial_state = get_initial_state()

    while True:
        first_run = (
            config.get("configurable", {}).get("thread_id") not in memory.storage
        )
        try:
            user_input = input("\n\nUser: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if first_run:
                initial_state["messages"].append(HumanMessage(content=user_input))
                state = initial_state
                # print("First run, state:", state)
            else:
                state = {"messages": [HumanMessage(content=user_input)]}
                # print("state:", state)
            workflow_compiled.invoke(state, config=config)
        except Exception as e:
            print(f"An error occurred. {e}")
            break
