import os
from typing import TypedDict
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

app = FastAPI()

# FreeLLMAPI Setup
os.environ["OPENAI_API_BASE"] = "https://freellmapi-tqja.onrender.com/v1"
os.environ["OPENAI_API_KEY"] = "freellmapi-01262c0df92253e89d91de4cb9865eabbe1cce879a27e4a6"

# Known stable models on FreeLLMAPI proxy
WORKING_MODELS = [
    "gemini-1.5-flash",
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gemini-2.5-flash"
]

def invoke_llm_with_retry(prompt: str):
    last_error = None
    for model_name in WORKING_MODELS:
        try:
            llm = ChatOpenAI(model=model_name, temperature=0.7, timeout=30)
            response = llm.invoke(prompt)
            if response and response.content:
                return response.content
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(f"All models failed. Last error: {last_error}")

# Define Graph State across agents
class MultiAgentState(TypedDict):
    topic: str
    research_notes: str
    critique: str
    final_output: str

# Node 1: Researcher Agent
def researcher_node(state: MultiAgentState):
    prompt = f"You are an expert researcher. Gather key facts and structured insights on: {state['topic']}."
    content = invoke_llm_with_retry(prompt)
    return {"research_notes": content}

# Node 2: Critic / Fact-Checker Agent
def critic_node(state: MultiAgentState):
    prompt = f"Review these research notes for accuracy and gaps. Suggest improvements:\n\n{state['research_notes']}"
    content = invoke_llm_with_retry(prompt)
    return {"critique": content}

# Node 3: Writer / Summarizer Agent
def writer_node(state: MultiAgentState):
    prompt = (
        f"Using the research notes and critique below, write a polished, professional response:\n\n"
        f"Research:\n{state['research_notes']}\n\n"
        f"Critique:\n{state['critique']}"
    )
    content = invoke_llm_with_retry(prompt)
    return {"final_output": content}

# Build StateGraph Multi-Agent Workflow
workflow = StateGraph(MultiAgentState)

# Add Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

# Connect Edges
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_edge("critic", "writer")
workflow.add_edge("writer", END)

app_graph = workflow.compile()

@app.get("/")
def home():
    return {"status": "Multi-Agent LangGraph System (Smart Model Failover) is Running 24/7!"}

@app.get("/run-graph")
def run_graph(topic: str = "Multi Agent AI Frameworks"):
    try:
        inputs = {"topic": topic}
        output = app_graph.invoke(inputs)
        return {
            "topic": topic,
            "final_output": output["final_output"],
            "research_notes": output["research_notes"],
            "critique": output["critique"]
        }
    except Exception as e:
        return {"error": str(e)}
