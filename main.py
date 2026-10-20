import os
from typing import TypedDict
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

app = FastAPI()

# FreeLLMAPI Setup with your custom URL and API Key
os.environ["OPENAI_API_BASE"] = "https://freellmapi-tqja.onrender.com/v1"
os.environ["OPENAI_API_KEY"] = "freellmapi-01262c0df92253e89d91de4cb9865eabbe1cce879a27e4a6"

llm = ChatOpenAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    topic: str
    result: str

def research_node(state: AgentState):
    prompt = f"Give 3 short key insights about {state['topic']}."
    response = llm.invoke(prompt)
    return {"result": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)

app_graph = workflow.compile()

@app.get("/")
def home():
    return {"status": "LangGraph Independent Service is Running 24/7!"}

@app.get("/run-graph")
def run_graph(topic: str = "AI Agents"):
    try:
        inputs = {"topic": topic}
        output = app_graph.invoke(inputs)
        return {"result": output["result"]}
    except Exception as e:
        return {"error": str(e)}
