import os
from typing import TypedDict
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

app = FastAPI()

# Groq Setup - Fetches API key securely from Render Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

class MultiAgentState(TypedDict):
    topic: str
    research_notes: str
    critique: str
    final_output: str

def researcher_node(state: MultiAgentState):
    prompt = f"You are an expert researcher. Gather key facts and structured insights on: {state['topic']}."
    response = llm.invoke(prompt)
    return {"research_notes": response.content}

def critic_node(state: MultiAgentState):
    prompt = f"Review these research notes for accuracy and gaps. Suggest improvements:\n\n{state['research_notes']}"
    response = llm.invoke(prompt)
    return {"critique": response.content}

def writer_node(state: MultiAgentState):
    prompt = (
        f"Using the research notes and critique below, write a polished, professional response:\n\n"
        f"Research:\n{state['research_notes']}\n\n"
        f"Critique:\n{state['critique']}"
    )
    response = llm.invoke(prompt)
    return {"final_output": response.content}

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
    return {"status": "Groq-Powered Multi-Agent LangGraph System is Running 24/7!"}

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
