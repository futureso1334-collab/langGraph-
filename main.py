import os
from typing import TypedDict
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

app = FastAPI()

# Current active Groq models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

def invoke_with_fallback(prompt: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing on Render.")
    
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.7)
            response = llm.invoke(prompt)
            if response and response.content:
                return response.content
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(f"All Groq models failed. Last error: {last_error}")

class MultiAgentState(TypedDict):
    topic: str
    research_notes: str
    critique: str
    final_output: str

def researcher_node(state: MultiAgentState):
    prompt = f"You are an expert researcher. Gather key facts and structured insights on: {state['topic']}."
    content = invoke_with_fallback(prompt)
    return {"research_notes": content}

def critic_node(state: MultiAgentState):
    prompt = f"Review these research notes for accuracy and gaps. Suggest improvements:\n\n{state['research_notes']}"
    content = invoke_with_fallback(prompt)
    return {"critique": content}

def writer_node(state: MultiAgentState):
    prompt = (
        f"Using the research notes and critique below, write a polished, professional response:\n\n"
        f"Research:\n{state['research_notes']}\n\n"
        f"Critique:\n{state['critique']}"
    )
    content = invoke_with_fallback(prompt)
    return {"final_output": content}

workflow = StateGraph(MultiAgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_edge("critic", "writer")
workflow.add_edge("writer", END)

app_graph = workflow.compile()

@app.get("/")
def home():
    return {"status": "Groq Multi-Agent System with Active Models is Running 24/7!"}

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
