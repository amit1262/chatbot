from shared.state import State
from langchain_core.prompts import SystemMessagePromptTemplate
from shared.language_model import language_model
from utils.utils import pretty_print_state
from langchain_core.messages import AIMessage


def generator(state: State):
    """
    Invoke the model with the current state.
    This function can be extended to include more complex logic.
    """
    chat_message = SystemMessagePromptTemplate.from_template(
        "{message}\n" "{context}",
    )
    message = state["messages"]
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    chat_message = chat_message.format(message=message, context=docs_content)

    if not chat_message:
        raise ValueError("No messages found in the state.")
    print(f"Agent: ")
    full_response = ""
    for chunk in language_model.stream([chat_message]):
        token = chunk.text()
        print(token, end="", flush=True)
        full_response += token

    return {"messages": [AIMessage(full_response)]}
