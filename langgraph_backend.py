import os
import sqlite3
from typing import Annotated, Literal, TypedDict
import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Add it to the .env file before starting the app."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    streaming=True,
)


@tool
def web_search(query: str) -> dict:
    """Search DuckDuckGo for a short factual answer."""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading") or query,
                    "summary": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                }
            )

        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "title": topic["Text"].split(" - ", 1)[0],
                        "summary": topic["Text"],
                        "url": topic.get("FirstURL", ""),
                    }
                )
            if len(results) >= 5:
                break

        return {"query": query, "results": results}
    except requests.RequestException as exc:
        return {"error": f"Search request failed: {exc}"}
    except ValueError:
        return {"error": "Search service returned an invalid response."}


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: Literal["add", "sub", "mul", "div"],
) -> dict:
    """Perform addition, subtraction, multiplication, or division."""
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    else:
        if second_num == 0:
            return {"error": "Division by zero is not allowed."}
        result = first_num / second_num

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest quote for a stock symbol using Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {
            "error": (
                "ALPHAVANTAGE_API_KEY is missing. Add it to the .env file "
                "to use stock-price lookup."
            )
        }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper().strip(),
                "apikey": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        quote = data.get("Global Quote")

        if not quote:
            return {
                "error": data.get("Note")
                or data.get("Information")
                or f"No quote was found for {symbol!r}."
            }

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "latest_trading_day": quote.get("07. latest trading day"),
        }
    except requests.RequestException as exc:
        return {"error": f"Stock-price request failed: {exc}"}
    except ValueError:
        return {"error": "Alpha Vantage returned an invalid response."}


TOOLS = [web_search, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(TOOLS)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """Ask Gemini to answer the user or request a tool call."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads() -> list[str]:
    """Return saved conversation IDs, newest checkpoints first."""
    thread_ids = {
        str(checkpoint.config["configurable"]["thread_id"])
        for checkpoint in checkpointer.list(None)
    }
    return sorted(thread_ids, reverse=True)
