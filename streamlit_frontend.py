import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_backend import chatbot, retrieve_all_threads


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat() -> None:
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": str(thread_id)}}
    )
    return state.values.get("messages", [])


def message_content_to_text(content) -> str:
    """Normalize plain text and Gemini content blocks for Streamlit."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                text_parts.append(str(block["text"]))
        return "".join(text_parts)
    return str(content) if content is not None else ""


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=f"thread-{thread_id}"):
        st.session_state["thread_id"] = str(thread_id)
        messages = load_conversation(str(thread_id))
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            content = message_content_to_text(message.content)
            if content:
                temp_messages.append({"role": role, "content": content})

        st.session_state["message_history"] = temp_messages


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        def ai_only_stream():
            for message_chunk, _metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage):
                    text = message_content_to_text(message_chunk.content)
                    if text:
                        yield text

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as exc:
            ai_message = f"Sorry, the request failed: {exc}"
            st.error(ai_message)

    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": str(ai_message)}
        )
