from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class State(TypedDict):
    """
    Define the state object for the chatbot.
    This can be extended with more fields as needed.
    """

    messages: Annotated[list, add_messages]
    context: list[Document]
