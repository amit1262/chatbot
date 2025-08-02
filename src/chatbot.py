from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
)
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings


class State(TypedDict):
    """
    Define the state of the chatbot.
    This can be extended with more fields as needed.
    """

    messages: Annotated[list, add_messages]
    context: list[Document]
    answer: str


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


def generator(state: State):
    """
    Invoke the model with the current state.
    This function can be extended to include more complex logic.
    """
    chat_message = ChatMessagePromptTemplate().from_template(
        "Use the System Instructions and the following pieces of retrieved context to generate a response to the user query."
        "Question: {message}\n",
        "Context: {context}",
    )
    message = state["messages"][-1].content
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    chat_message = chat_message.format(message=message, context=docs_content)

    if not chat_message:
        raise ValueError("No messages found in the state.")
    print(f"Agent: ")
    full_response = ""
    for chunk in language_model.stream(chat_message):
        token = chunk.text()
        print(token, end="", flush=True)
        full_response += token
    print("\n")
    print("*" * 100)
    print("\n")

    return {"messages": [AIMessage(content=full_response)]}


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


def initialize_language_model():
    """
    Initialize the model using environment variables.
    This function can be extended to include more complex logic.
    """
    # model initialization
    MODEL_NAME = os.environ.get("LANGUAGE_MODEL_NAME")
    try:
        model = init_chat_model(
            model=MODEL_NAME, model_provider=os.environ.get("LANGUAGE_MODEL_PROVIDER")
        )
        assert model, f"Failed to initialize {MODEL_NAME} model"
        print(f"{MODEL_NAME} initialized successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error initializing model {MODEL_NAME}: {e}")


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


def get_vector_index(index_params: dict):
    """Generates a vector index from the provided documents.

    Args:
        index_params (dict): dict of parameters for indexing

    Returns:
        FAISS: index of the documents.
    """
    INDEX_OUTPUT_PATH = index_params.get("output_path")
    # embedding model
    EMBEDDING_MODEL_NAME = index_params.get("emb_model")
    model = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Check if the vector index already exists
    if os.path.exists(INDEX_OUTPUT_PATH):
        print("Loading existing vector index...")
        vector_index = FAISS.load_local(INDEX_OUTPUT_PATH, embeddings=model)
    else:
        print(f"Creating a new vector index at: {INDEX_OUTPUT_PATH}\n")
        # get chunks of text from the documents
        CHUNK_SIZE = index_params.get("chunk_size", 1000)
        CHUNK_OVERLAP = index_params.get("chunk_overlap", 200)
        SRC_DOC_PATHS = index_params.get("doc_paths")
        # Load the PDF documents
        docs = []
        for path in SRC_DOC_PATHS:
            try:
                loader = PyMuPDFLoader(path)
                docs.extend(loader.load())
            except Exception as e:
                raise RuntimeError(f"Error loading document {path}: {e}")

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)

        vector_index = FAISS.from_documents(chunks, model)
        vector_index.save_local(INDEX_OUTPUT_PATH)
        print(f"Vector index created and saved successfully at: {INDEX_OUTPUT_PATH}.")

    return vector_index


def retriever(state):
    """Get context from the vector index based on the user's query.

    Args:
        state (_type_): current state of the chatbot.

    Returns:
        _type_: context retrieved from the vector index.
    """
    context = vector_index.similarity_search(query=state["messages"][-1].content, k=3)
    return {"context": context}


if __name__ == "__main__":
    # initialize the chatbot
    load_dotenv()
    language_model = initialize_language_model()
    index_params = {
        "output_path": "../data/vector_index/ind",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "doc_paths": [
            "../docs/OHDSI/TheBookOfOhdsi.pdf",
            "../docs/Strategus/Strategus.pdf",
            "../docs/Strategus/ExecuteStrategus.pdf",
            "../docs/Strategus/IntroductionToStrategus.pdf",
            "../docs/Strategus/CreatingAnalysisSpecification.pdf",
        ],
        "emb_model": os.environ.get("EMBEDDING_MODEL_NAME"),
    }
    # create vector index
    vector_index = get_vector_index(index_params=index_params)

    print("*" * 100)

    # workflow graph definition
    workflow = StateGraph(state_schema=State)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "bot_3"}}
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
                # print("First run, state:", state)
            else:
                state = {"messages": [HumanMessage(content=user_input)]}
            print("-" * 100)
            print(f"Current State: {state}\n")
            print("-" * 100)
            workflow_compiled.invoke(state, config=config)
        except Exception as e:
            print(f"An error occurred. {e}")
            break

        first_run = (
            config.get("configurable", {}).get("thread_id") not in memory.storage
        )
