from dotenv import load_dotenv

load_dotenv()
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.state import State
from utils.utils import pretty_print_state
from graph_nodes.retriever import retriever
from graph_nodes.generator import generator

# from graph_nodes.prompt_builder import prompt_builder


def get_initial_state():
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


def get_user_input(current_state: State):
    """
    Get user input and update the state.
    This function can be extended to include more complex logic.
    """
    user_input = input("\n\nUser: ").strip()
    if not user_input:
        raise ValueError("User input cannot be empty.")

    # Append the user message to the current state
    return {"messages": [HumanMessage(content=user_input)]}


def prepare_documents_for_indexing() -> list:
    """Load and prepare documents for indexing.

    Returns:
        list: List of documents split into chunks.
    """
    # parameters for text splitting
    CHUNK_SIZE = 1000  # characters per chunk
    CHUNK_OVERLAP = 200

    # Load the PDF documents
    docs = []
    doc_paths = ["../docs/OHDSI/TheBookOfOhdsi.pdf"]
    for path in doc_paths:
        try:
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading document {path}: {e}")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":
    # initialize the chatbot
    print("*" * 100)

    # workflow graph definition
    workflow = StateGraph(state_schema=State)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "bot_5"}}
    workflow_compiled = workflow.compile(checkpointer=memory)

    # workflow invocation
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
            else:
                state = {"messages": [HumanMessage(content=user_input)]}
            # pretty_print_state(state)
            state = workflow_compiled.invoke(state, config=config)
            pretty_print_state(state)
        except Exception as e:
            raise RuntimeError(f"Error workflow execution: {e}")
            break

        first_run = (
            config.get("configurable", {}).get("thread_id") not in memory.storage
        )
