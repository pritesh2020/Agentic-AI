from langchain.tools import tool
from geopy.geocoders import Nominatim
import requests
from typing import Dict
import re
import emoji


geolocator = Nominatim(user_agent="orchestrator")
@tool("get_weather", return_direct=True)
def get_weather(city: str) -> str:
    """Return simple weather for a given city from a tiny offline database.
    Input: city name (e.g., 'Paris')."""
    c = city.strip().lower().replace('"', '').replace("'", "")
    location = geolocator.geocode(c)
    if location:
        c = location.address.split(",")[0]
    # remove quotes, extra spaces, lowercase

    print(f"Debug: Looking up weather for {c}")
    url = f"http://wttr.in/{c}?format=3"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        weather = response.text.strip()
        # remove all special chars.
        weather = emoji.demojize(weather)
        weather = ''.join(e for e in weather if e.isalnum() or e.isspace() or e in ['C', 'F', '%', 'm', 'h', 'p','/',':','-','+'])
        return weather
    return f"No weather found for '{c}'. Assume mild (22°C) and clear for demo."



@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Safely evaluate a simple math expression. Accepts raw text like
    '23*17 + 3.5' or lines such as 'expression = "23*17 + 3.5"'."""
    import math, re

    s = expression.strip()
    # Accept patterns like: expression = "...", expr: ..., calc="..."
    m_eq = re.match(r"^[A-Za-z_][\w\-]*\s*(?:=|:)\s*(.*)$", s)
    if m_eq:
        s = m_eq.group(1).strip()
    # Strip surrounding quotes/backticks if present
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in "'\"`"):
        s = s[1:-1].strip()

    # Basic validation: digits, ops, spaces, parentheses, decimal dot, percent
    if not re.fullmatch(r"[0-9+\-*/(). %\s]+", s):
        return "Calculator error: invalid characters."
    try:
        # Evaluate in a safe namespace (only math allowed)
        result = eval(s, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"



@tool("mini_wiki", return_direct=True)
def mini_wiki(topic: str) -> str:
    """Return a short summary from a tiny offline encyclopedia.
    Useful for quick references when internet is unavailable."""
    kb: Dict[str, str] = {
        "alan turing": "Alan Turing (1912–1954) was a mathematician and pioneer of computer science who formalized computation and contributed to codebreaking in WWII.",
        "agentic ai": "Agentic AI refers to systems that can plan, choose tools/actions, and adapt using feedback, rather than only producing text responses.",
        "langchain": "LangChain is a framework that provides building blocks for LLM apps: prompts, chains, tools, memory, and agents.",
    }
    topic = topic.strip().lower().replace('"', '').replace("'", "")
    print(f"Debug: Looking up mini_wiki for '{topic}'")
    return kb.get(topic, "No entry found in mini_wiki. Try 'Alan Turing', 'Agentic AI', or 'LangChain'.")


def _parse_city_weather(query: str):
    """Extract city and optional weather from a single string.
    Accepts: "Paris" OR "city=Paris" OR "city=Paris; weather=sunny, 24°C"
    Returns: (city, weather) with weather possibly ''.
    """
    q = query.strip()
    if ";" in q or "city=" in q.lower():
        parts = [p.strip() for p in q.split(";")]
        city, weather = "", ""
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower(); v = v.strip()
                if k == "city":
                    city = v
                elif k == "weather":
                    weather = v
        if not city:
            city = re.split(r"[;,]", q)[0].replace("city", "").replace("=", "").strip()
        return city, weather
    return q, ""

@tool("suggest_city_activities", return_direct=False)
def suggest_city_activities(query: str) -> str:
    """Recommend ONE indoor and ONE outdoor activity.
    Input: SINGLE string. Examples:
      - "Paris"
      - "city=Paris"
      - "city=Paris; weather=sunny, 24°C"
    """
    catalog = {
        "chicago": {
            "indoor": ["Art Institute of Chicago", "Museum of Science and Industry", "Field Museum"],
            "outdoor": ["Chicago Riverwalk", "Millennium Park", "Navy Pier"],
        },
        "paris": {
            "indoor": ["Louvre Museum", "Musée d'Orsay"],
            "outdoor": ["Seine River Walk", "Jardin du Luxembourg"],
        },
        "london": {
            "indoor": ["British Museum", "Tate Modern"],
            "outdoor": ["Hyde Park", "South Bank Walk"],
        },
        "tokyo": {
            "indoor": ["teamLab Planets", "Tokyo National Museum"],
            "outdoor": ["Ueno Park", "Shibuya Crossing Walk"],
        },
        "mumbai": {
            "indoor": ["Chhatrapati Shivaji Maharaj Vastu Sangrahalaya", "Phoenix Mall"],
            "outdoor": ["Marine Drive", "Sanjay Gandhi National Park"],
        },
    }
    city, weather = _parse_city_weather(query)
    c = city.strip().strip("'\"`").lower()
    if not c:
        return "Please provide a city name (e.g., 'city=Paris; weather=sunny, 24°C')."
    data = catalog.get(c)
    if not data:
        return ("General suggestions (city not in catalog): Indoor - visit a local museum or aquarium. "
                "Outdoor - take a riverfront/park walk if conditions allow.")
    w = weather.lower()
    indoor_first = any(k in w for k in ["rain", "storm"]) or ("overcast" in w and "cold" in w)
    if indoor_first:
        indoor = data["indoor"][0]; outdoor = data["outdoor"][0]
    elif any(k in w for k in ["sunny", "clear"]):
        outdoor = data["outdoor"][0]; indoor = data["indoor"][0]
    else:
        indoor = data["indoor"][0]; outdoor = data["outdoor"][0]
    return f"City: {city}. Indoor: {indoor}. Outdoor: {outdoor}. (Weather-aware heuristics.)"
