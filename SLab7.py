# SLab7.py
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import Tools
import model_selector

# (Optional) Silence LangChain deprecation warnings in this learning lab
try:
    from langchain_core._api import LangChainDeprecationWarning
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass


# -----------------------------
# Tools (single-input where required for ReAct safety)
# -----------------------------

# -----------------------------
# LLM and Tool Registry
# -----------------------------
llm = model_selector.get_models("ollama")  # or "openai"
# llm.invoke("Hello, how are you?")
# llm = model_selector.get_models("openai")

tools = [Tools.get_weather, Tools.calculator, Tools.mini_wiki, Tools.suggest_city_activities]

# -----------------------------
# 1) Tool-Calling Agent
# -----------------------------
tool_calling_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "system_message": (
            "You are a helpful english assistant. Use tools when relevant. Send only english text to tools. Finish with 'Final Answer:' and a concise result."
        )
    }
)

# -----------------------------
# 2) ReAct Agent (Zero-Shot)
# -----------------------------
react_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": (
            "You are a helpful assistant. Think step-by-step, then use tools. "
            "Use ONLY these section headers: Thought:, Action:, Action Input:, Observation:. "
            "End your final response with a line starting 'Final Answer:'."
        )
    }
)

# -----------------------------
# 3) Conversational Agent with Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conversational_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    agent_kwargs={
        "prefix": (
            "You are a friendly assistant that remembers prior messages in this chat. "
            "Use tools as needed. End your final response with 'Final Answer:'."
        )
    }
)

# -----------------------------
# Demos
# -----------------------------

def demo_tool_calling():
    print("\n=== Demo 1: Tool-Calling Agent ===\n")
    q = "What's the weather in Tokyo and suggest one indoor and one outdoor thing to do this evening."
    print("User:", q)
    r = tool_calling_agent.invoke({"input": q})
    print("Assistant:", r["output"])

def demo_react():
    print("\n=== Demo 2: ReAct Agent ===\n")
    q = "Calculate: 23*17 + 3.5, then check mini_wiki for LangChain and summarize both in one line."
    print("User:", q)
    r = react_agent.invoke({"input": q})
    print("Assistant:", r["output"])

def demo_conversational_memory():
    print("\n=== Demo 3: Conversational Agent with Memory ===\n")
    turns = [
        "Hi, my name is Shobhit",
        "Plan a relaxed evening for me in Paris. Remember my name.",
        "What was my name? Also suggest one indoor and one outdoor activity."
    ]
    for t in turns:
        print("\nUser:", t)
        r = conversational_agent.invoke({"input": t})
        print("Assistant:", r["output"])

if __name__ == "__main__":
    demo_tool_calling()
    demo_react()
    demo_conversational_memory()