# LangGraph Gemini Chatbot with Memory

> A beginner-friendly Python chatbot built with LangGraph, LangChain, Google Gemini 2.5 Flash, and Streamlit.

## 📌 Project Description

**LangGraph Gemini Chatbot with Memory** is a conversational AI application that demonstrates how to build a multi-turn chatbot using LangGraph's graph-based workflow system.

The project uses Google's **Gemini 2.5 Flash** model through LangChain and stores conversational context using LangGraph's `InMemorySaver`. Each conversation is associated with a unique `thread_id`, allowing the chatbot to remember previous messages within the same session.

This project is suitable for learning:

- LangGraph state management
- Conversational memory with checkpoints
- Gemini integration through LangChain
- Streamlit chatbot UI development
- Secure API key loading with `.env`
- Python virtual environment workflows

## ✨ Features

- 🤖 Chatbot powered by **Gemini 2.5 Flash**
- 🧠 Conversational memory using LangGraph `InMemorySaver`
- 🔁 Supports multi-turn conversations
- 🧵 Conversation history tracked with a `thread_id`
- 🔐 API keys loaded securely from a `.env` file
- 🧩 Graph-based chatbot workflow using `StateGraph`
- 💬 Streamlit chat interface
- 🐍 Python virtual environment support

## 🏗️ Project Architecture

```text
User
 |
 v
Streamlit Frontend
 |
 v
LangGraph
 |
 v
chat_node
 |
 v
Gemini 2.5 Flash
 |
 v
Response
 |
 v
InMemorySaver (Memory)
```

### Flow Summary

1. The user enters a message in the Streamlit chat interface.
2. The frontend sends the message to the compiled LangGraph chatbot.
3. LangGraph passes the conversation state to `chat_node`.
4. `chat_node` calls Gemini 2.5 Flash using LangChain.
5. Gemini returns a response.
6. LangGraph stores the updated message history using `InMemorySaver`.
7. The response is displayed in the Streamlit UI.

## 🛠️ Technologies Used

- **Python** - Core programming language
- **LangGraph** - Graph-based AI workflow orchestration
- **LangChain** - LLM integration framework
- **Google Gemini 2.5 Flash** - Generative AI model
- **Streamlit** - Web-based chatbot frontend
- **python-dotenv** - Environment variable management
- **Virtual Environment** - Dependency isolation

## 📁 Folder Structure

```text
Lang-graph-chatbot/
|
+-- langgraph_backend.py      # LangGraph workflow, Gemini setup, and memory
+-- streamlit_frontend.py     # Streamlit chat interface
+-- requirements.txt          # Python dependencies
+-- .env                      # Environment variables, not committed to Git
+-- .gitignore                # Git ignored files and folders
+-- README.md                 # Project documentation
+-- .venv/                    # Local virtual environment
```

## ⚙️ Installation Guide

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Lang-graph-chatbot
```

If you already have the project locally, open the project folder in your terminal.

## 🐍 Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should show the virtual environment name, usually `(.venv)`.

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔐 Configure Environment Variables

Create a `.env` file in the project root directory.

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Replace `your_google_api_key_here` with your actual Google API key.

> Do not commit your `.env` file to GitHub. It contains private credentials.

## ▶️ How to Run the Project

Start the Streamlit application:

```bash
streamlit run streamlit_frontend.py
```

After running the command, Streamlit will open the app in your browser.

Default local URL:

```text
http://localhost:8501
```

## 🧠 Code Explanation

### StateGraph

`StateGraph` is the core LangGraph component used to define the chatbot workflow.

In this project, the graph has:

- A `START` point
- One processing node called `chat_node`
- An `END` point

```python
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
```

This means every user message flows through `chat_node`, receives a Gemini response, and then finishes the graph execution.

### ChatState

`ChatState` defines the structure of the chatbot's state.

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `messages` field stores the conversation history. The `add_messages` reducer tells LangGraph how to append new messages to the existing conversation.

### chat_node

`chat_node` is the function that receives the current conversation state and sends the messages to Gemini.

```python
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}
```

It performs three main tasks:

- Reads the current messages from the state
- Sends those messages to the Gemini model
- Returns the AI response so LangGraph can add it to the state

### InMemorySaver

`InMemorySaver` is LangGraph's in-memory checkpointing system.

```python
checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)
```

It stores conversation state while the application is running. This allows the chatbot to remember earlier messages in the same conversation thread.

> Since this memory is in-memory, it resets when the application restarts.

### thread_id

The `thread_id` identifies a specific conversation thread.

```python
CONFIG = {'configurable': {'thread_id': 'thread-1'}}
```

LangGraph uses this value to know which saved conversation history should be loaded and updated. Different `thread_id` values can represent different conversations or users.

### Gemini Integration

Gemini is connected through LangChain's `ChatGoogleGenerativeAI`.

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

The API key is loaded from the `.env` file using `python-dotenv`.

```python
load_dotenv()
```

This keeps sensitive credentials out of the source code.

## 💬 Example Conversation

```text
User:
Hi, my name is Vivek.

Assistant:
Hello Vivek! How can I help you today?

User:
What is my name?

Assistant:
Your name is Vivek.
```

The chatbot can answer the second question because the previous message is preserved in the conversation state for the active `thread_id`.

## 🚀 Future Improvements

- Add persistent memory using SQLite, PostgreSQL, or Redis
- Generate a unique `thread_id` for every user session
- Add a sidebar to manage multiple conversations
- Add streaming responses from Gemini
- Improve UI styling and layout
- Add deployment instructions for Streamlit Community Cloud
- Add error handling for missing or invalid API keys
- Add unit tests for the LangGraph workflow

## 🧯 Troubleshooting

### `GOOGLE_API_KEY` is missing

Make sure your `.env` file exists in the project root and contains:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### Streamlit command not found

Install the dependencies again:

```bash
pip install -r requirements.txt
```

Also make sure your virtual environment is activated.

### Chatbot does not remember previous messages

Check that the same `thread_id` is being used across requests:

```python
CONFIG = {'configurable': {'thread_id': 'thread-1'}}
```

Also remember that `InMemorySaver` clears memory when the app restarts.

### Gemini API error

Verify that:

- Your API key is valid
- The `.env` file is correctly configured
- Your Google account has access to the Gemini API
- The model name is correct: `gemini-2.5-flash`

## 📄 License

This project is available for educational and portfolio use.

You may add a specific license such as MIT by creating a `LICENSE` file in the project root.
