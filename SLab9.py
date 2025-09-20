from typing import Dict, Tuple
import re
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

import Tools
import model_selector

llm = model_selector.get_models("ollama")  # or "openai"
# llm.invoke("Hello, how are you?")
# llm = model_selector.get_models("openai")

TOOLS = [Tools.get_weather, Tools.calculator, Tools.mini_wiki, Tools.suggest_city_activities]

class AgentState(TypedDict):
    # ACCUMULATE messages across nodes in order (prevents tool role error)
    messages: Annotated[list[AnyMessage], add]

llm = model_selector.get_models("ollama")  # or "openai"
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> AgentState:
    """Call the chat model. If tool calls are returned, the router will send us to Tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)  # routes to tools or END
graph.add_edge("tools", "agent")     



# Persistence: in-memory checkpointer (swap with SQLite/Redis for prod)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)



def run_single_turn():
    print("\n=== Single-turn demo ===")
    cfg = {"configurable": {"thread_id": "t-single"}}
    out = app.invoke({"messages": [HumanMessage(content="Plan an evening in Chicago with one indoor and one outdoor, based on the weather.")]}, cfg)
    final_msg = out["messages"][-1]
    print("Assistant:", getattr(final_msg, "content", final_msg))

def run_multi_turn_with_persistence():
    print("\n=== Multi-turn demo with persistence (threaded) ===")
    cfg = {"configurable": {"thread_id": "t-123"}}  # same thread reuses memory
    msgs = [
        HumanMessage(content="Hi, my name is Priya."),
        HumanMessage(content="Please plan a relaxed evening for me in Paris. Remember my name."),
        HumanMessage(content="What was my name? Also suggest one indoor and one outdoor activity."),
    ]
    for m in msgs:
        out = app.invoke({"messages": [m]}, cfg)
        final_msg = out["messages"][-1]
        print("\nUser:", m.content)
        print("Assistant:", getattr(final_msg, "content", final_msg))

def run_math_tool():
    print("\n=== Tool routing for math (calculator) ===")
    cfg = {"configurable": {"thread_id": "t-math"}}
    out = app.invoke({"messages": [HumanMessage(content="Compute 23*17 + 3.5 and return only the number.")]}, cfg)
    final_msg = out["messages"][-1]
    print("Assistant:", getattr(final_msg, "content", final_msg))

if __name__ == "__main__":
    run_single_turn()
    run_multi_turn_with_persistence()
    run_math_tool()
