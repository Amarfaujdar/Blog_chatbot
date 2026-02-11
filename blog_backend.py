from __future__ import annotations

import json
import re
import os
import operator
import base64
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Any

# --- Pydantic V1 for LangChain Compatibility ---
from pydantic.v1 import BaseModel, Field, root_validator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# --- LangChain & Ollama ---
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()  

# ==========================================
# 0. HELPER: Clean JSON from Markdown
# ==========================================
def clean_json_output(text: str) -> str:
    """Robustly extracts the first JSON object from the text."""
    text = text.strip()
    # Try to find the JSON block using regex (non-greedy match for the outer braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

# ==========================================
# 1. ROBUST SCHEMAS (Pydantic V1)
# ==========================================

class Task(BaseModel):
    id: int = Field(default=0)
    title: str = Field(default="Section")
    goal: str = Field(..., description="Section goal")
    bullets: List[str] = Field(..., min_length=1)
    target_words: int = Field(..., description="Word count")
    
    @root_validator(pre=True)
    def map_deepseek_task_keys(cls, values: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "section_title": "title", "section_goal": "goal",
            "content": "bullets", "target_word_count": "target_words"
        }
        for old_key, new_key in mapping.items():
            if old_key in values:
                values[new_key] = values.pop(old_key)
        return values

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[Task]

    @root_validator(pre=True)
    def fix_deepseek_plan(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "blog_plan" in values: return values["blog_plan"]
        if "plan" in values and isinstance(values["plan"], list):
            return {
                "blog_title": "DeepSeek Generated Blog",
                "audience": "Developers",
                "tone": "Technical",
                "tasks": values["plan"]
            }
        return values

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

    @root_validator(pre=True)
    def fix_deepseek_router(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "router_decision" in values:
            values = values["router_decision"]
            
        if "mode" not in values:
            values["mode"] = "hybrid" if values.get("needs_research") else "closed_book"

        if values.get("mode") in ["hybrid", "open_book"]:
            values["needs_research"] = True
            if not values.get("queries"):
                values["queries"] = ["latest trends and best practices"]
        return values

# --- Diagram Schemas ---
class DiagramSpec(BaseModel):
    target_section: str = Field(..., description="The exact section header (e.g. '## Section Name') to insert the diagram after")
    type: Literal["flowchart", "sequence", "class", "state", "er", "mindmap", "pie", "gantt", "quadrantChart", "requirement", "gitGraph"]
    code: str = Field(..., description="Raw Mermaid code without markdown backticks")
    caption: str

class GlobalDiagramPlan(BaseModel):
    diagrams: List[DiagramSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[dict]
    plan: Optional[dict]
    sections: Annotated[List[tuple], operator.add] 
    merged_md: str
    # md_with_placeholders removed
    diagram_specs: List[dict]
    final: str

# ==========================================
# 2. LLM SETUP
# ==========================================
llm = ChatOllama(
    model="deepseek-v3.1:671b-cloud", 
    temperature=0,
)

# ==========================================
# 3. ROUTER NODE
# ==========================================
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Your task is to decide whether web research is required BEFORE planning the content.

Topic classification rules:

- closed_book (needs_research = false):
  Use this when the topic is evergreen and correctness does not depend on recent facts
  (e.g., core concepts, fundamentals, theory, definitions).

- hybrid (needs_research = true):
  Use this when the topic is mostly evergreen but benefits from up-to-date examples,
  tools, models, libraries, or best practices.

- open_book (needs_research = true):
  Use this when the topic is highly time-sensitive or volatile
  (e.g., “latest”, “this week”, “recent updates”, rankings, pricing, policies, regulations).

If needs_research = true:
- Generate 3–10 high-signal, specific web search queries.
- Queries must be scoped and actionable (avoid generic terms like just “AI” or “LLM”).
- If the user mentions time constraints like “latest”, “this week”, or “last week”,
  those constraints MUST be reflected directly in the queries.

Return STRICT JSON only (no markdown, no explanation):

{
  "needs_research": boolean,
  "mode": "closed_book" | "hybrid" | "open_book",
  "queries": ["query1", "query2"]
}
"""


def router_node(state: State) -> dict:
    print(f"--- Router Node (Topic: {state['topic']}) ---")
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    
    try:
        clean_text = clean_json_output(response.content)
        data = json.loads(clean_text)
        decision = RouterDecision(**data)
        print(f"Mode: {decision.mode} | Research Required: {decision.needs_research}")
        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries,
        }
    except Exception as e:
        print(f"Router Error: {e}. Defaulting to Hybrid.")
        return {"needs_research": True, "mode": "hybrid", "queries": [f"{state['topic']} trends"]}

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# ==========================================
# 4. RESEARCH NODE
# ==========================================
def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    print(f"  > Searching Tavily: {query}")
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        normalized = []
        for r in results or []:
            normalized.append({
                "title": r.get("title", "No Title"),
                "url": r.get("url", ""),
                "snippet": r.get("content", "") or r.get("snippet", ""),
            })
        return normalized
    except Exception as e:
        print(f"    Tavily Error: {e}")
        return []

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Your task is to synthesize raw web search results into a clean, deduplicated list of evidence items.

Evidence selection rules:
- Only include results with a non-empty URL.
- Prefer relevant and authoritative sources (official documentation, company blogs, reputable publications).
- Deduplicate strictly by URL.
- Keep snippets concise and informative.
- Do NOT invent or infer information.

Output rules:
- Return STRICT JSON only (no markdown, no explanations).
- The output must exactly match the following schema.

{
  "evidence": [
    { "title": "...", "url": "...", "snippet": "..." }
  ]
}
"""


def research_node(state: State) -> dict:
    queries = state.get("queries", [])[:3]
    raw_results = []
    
    for q in queries:
        raw_results.extend(_tavily_search(q))

    if not raw_results:
        return {"evidence": []}

    print("  > Synthesizing evidence...")
    response = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw Results:\n{str(raw_results)[:10000]}"),
        ]
    )

    try:
        clean_text = clean_json_output(response.content)
        data = json.loads(clean_text)
        items = data if isinstance(data, list) else data.get("evidence", [])
        return {"evidence": items}
    except Exception as e:
        print(f"  > Research Parse Error: {e}")
        return {"evidence": []}

# ==========================================
# 5. ORCHESTRATOR NODE
# ==========================================
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.

Your task is to create a highly actionable outline for a technical blog post.

Planning requirements:
- Create 5–9 sections (tasks) appropriate for the topic and target audience.
- Each task MUST include:
  1) title
  2) goal (exactly 1 sentence)
  3) 3–6 concrete, specific, non-overlapping bullets
  4) target word count between 120–550 words

Quality bar:
- Assume the reader is a developer; use correct technical terminology.
- Bullets must be actionable (e.g., build, compare, measure, verify, debug).
- Across the full plan, include at least TWO of the following (where relevant):
  - minimal code sketch / MWE
  - edge cases or failure modes
  - performance or cost considerations
  - security or privacy considerations
  - debugging or observability tips

Grounding rules by mode:
- closed_book:
  - Keep content evergreen.
  - Do NOT rely on external evidence or recent facts.
- hybrid:
  - Use evidence for up-to-date tools, models, or releases.
  - Sections that rely on fresh info should reflect that in their bullets.
- open_book:
  - Focus on summarizing recent events and implications.
  - Avoid tutorial or step-by-step sections unless explicitly requested.
  - If evidence is insufficient, create a transparent plan noting limited sources.

Output rules:
- Return STRICT JSON only (no markdown, no explanations).
- The output must exactly match the following schema.

{
  "blog_title": "...",
  "audience": "...",
  "tone": "...",
  "tasks": [
    {
      "title": "...",
      "goal": "...",
      "bullets": ["..."],
      "target_words": 200
    }
  ]
}
"""


def orchestrator_node(state: State) -> dict:
    print("--- Orchestrator Node ---")
    evidence = state.get("evidence", [])
    
    response = llm.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Evidence: {[e.get('title') for e in evidence[:5]]}"
                )
            ),
        ]
    )
    
    try:
        clean_text = clean_json_output(response.content)
        data = json.loads(clean_text)
        plan = Plan(**data)
        return {"plan": plan.dict()}
    except Exception as e:
        print(f"Plan Parse Error: {e}")
        fallback = Plan(
            blog_title=state["topic"], audience="Devs", tone="Tech", 
            tasks=[Task(id=1, title="Intro", goal="Explain", bullets=["Point 1"], target_words=200)]
        )
        return {"plan": fallback.dict()}

# ==========================================
# 6. FANOUT & WORKER
# ==========================================
def fanout(state: State):
    tasks_with_ids = []
    if state["plan"]:
        plan_data = state["plan"] 
        tasks = plan_data.get("tasks", [])
        for i, task_data in enumerate(tasks):
            if "id" not in task_data or task_data["id"] == 0:
                task_data["id"] = i + 1
            tasks_with_ids.append(task_data)

    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "plan": state["plan"],
                "evidence": state.get("evidence", []),
            },
        )
        for task in tasks_with_ids
    ]

# --- UPDATED: Worker System Prompt (Encourages Tables) ---
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.

Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover all bullets.
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Start with a '## <Section Title>' heading.
- Use Markdown tables for comparisons or data if appropriate.
- Output ONLY the section content in Markdown.
- Do NOT output JSON.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = payload.get("evidence", [])
    
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join([f"- {e.get('title')} ({e.get('url')})" for e in evidence[:10]])

    print(f"Writing Section: {task.title}")

    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Section: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Bullets:{bullets_text}\n"
                    f"Evidence:\n{evidence_text}\n"
                )
            ),
        ]
    )
    return {"sections": [(task.id, response.content.strip())]}

# ============================================================
# 7. REDUCER SUBGRAPH
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered = [text for _, text in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    title = plan.get("blog_title") if plan else "Blog Post"

    # --- Append Sources ---
    evidence = state.get("evidence", [])
    seen_urls = set()
    unique_sources = []
    
    for item in evidence:
        url = item.get("url")
        title_source = item.get("title", "Source")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(f"- [{title_source}]({url})")
            
    sources_section = ""
    if unique_sources:
        sources_section = "\n\n## Sources\n\n" + "\n".join(unique_sources)
    
    merged_md = f"# {title}\n\n{body}\n{sources_section}"
    return {"merged_md": merged_md}

# --- UPDATED: Diagram Decision Prompt (Strict Max 2) ---
DECIDE_DIAGRAMS_SYSTEM = """You are a Technical Editor.
Identify complex concepts that need visualization (logic flows, system architecture).

Rules:
1. STRICT LIMIT: Maximum 2 diagrams total.
2. Select the EXACT section header (e.g. '## 1. Overview') where the diagram provides the most value.
3. SUPPORTED TYPES: flowchart, sequence, class, state, er, mindmap, pie, gantt, quadrantChart.
4. SYNTAX SAFETY:
   - For 'mindmap', use STRICT indentation (2 spaces) and surround all node text with quotes if it has spaces or symbols. Example: `id["Node Text"]`
   - For 'flowchart', use alphanumeric node IDs (A, B) and put text in brackets/quotes. A["Text"]
   - Do NOT use Markdown syntax (bold **, italics *) inside the diagram code.
5. Return STRICT JSON:
{
  "diagrams": [
    { 
      "target_section": "## 2. Architecture", 
      "type": "flowchart", 
      "code": "graph TD\n  A[\"Start\"] --> B{\"Check\"}\n  B -- Yes --> C[\"Done\"]", 
      "caption": "Figure 1: Process Flow" 
    }
  ]
}
Note: 'code' must be raw mermaid syntax. Do not wrap 'code' in markdown backticks in JSON.
"""

def decide_diagrams(state: State) -> dict:
    print("--- Deciding Diagrams (Mermaid) ---")
    merged_md = state["merged_md"]
    
    response = llm.invoke(
        [
            SystemMessage(content=DECIDE_DIAGRAMS_SYSTEM),
            HumanMessage(content=f"Text to review:\n{merged_md[:15000]}"),
        ]
    )

    print(f"  > Raw LLM Response (First 100 chars): {response.content[:100]}...")

    try:
        clean_text = clean_json_output(response.content)
        data = json.loads(clean_text)
        diagram_plan = GlobalDiagramPlan(**data)
        
        # Enforce limit in code just in case LLM hallucinations
        final_diagrams = diagram_plan.diagrams[:2]
        
        print(f"  > Generated {len(final_diagrams)} diagrams.")
        return {
            "diagram_specs": [d.dict() for d in final_diagrams],
        }
    except Exception as e:
        print(f"Diagram Decision Failed: {e}. Proceeding without diagrams.")
        return {
            "diagram_specs": []
        }

def inject_diagrams(state: State) -> dict:
    specs = state.get("diagram_specs", [])
    md = state.get("merged_md", "")

    if not specs:
        return {"final": md}

    print("--- Injecting Diagrams into Markdown ---")
    
    final_md = md
    
    for spec in specs:
        target = spec["target_section"]
        code = spec["code"]
        caption = spec["caption"]
        
        # CLEANING: Remove potential markdown fences or "mermaid" keyword inside the code
        # LLMs often output: ```mermaid\ngraph TD...``` or just `mermaid\ngraph TD...`
        code = code.replace("```mermaid", "").replace("```", "").strip()
        if code.startswith("mermaid"):
            code = code[7:].strip()

        # BASE64 ENCODING STRATEGY
        # Bypassing Markdown parsers completely.
        # We encode the code to base64, stick it in a data attribute.
        # Frontend will decode and render.
        code_b64 = base64.b64encode(code.encode('utf-8')).decode('utf-8')
        
        mermaid_block = (
            f"\n\n#### {caption}\n"
            f'<div class="mermaid-encoded" data-code="{code_b64}"></div>\n'
        )
        
        # Robust Injection Strategy:
        # 1. Try exact match
        if target in final_md:
            final_md = final_md.replace(target, f"{target}\n{mermaid_block}")
            print(f"  > Injected {spec['type']} (Exact) after {target}")
            continue

        # 2. Try fuzzy match (ignore case, spaces, and '##')
        # We look for lines in final_md that 'look like' the target
        import re
        lines = final_md.split('\n')
        injected = False
        
        # Prepare target core: lowercase, removed non-alphanumeric
        target_core = re.sub(r'[^a-z0-9]', '', target.lower())
        
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                line_core = re.sub(r'[^a-z0-9]', '', line.lower())
                # specific check: if target is just "Overview" and line is "## Overview" -> match
                # or if target is "## 1. Overview" and line is "## Overview" -> match (common mismatch)
                if target_core in line_core or line_core in target_core:
                     # Reconstruct the text with injection
                     lines[i] = f"{line}\n{mermaid_block}"
                     final_md = "\n".join(lines)
                     print(f"  > Injected {spec['type']} (Fuzzy) after {line.strip()}")
                     injected = True
                     break
        
        if not injected:
            # Fallback: Append to end if target not found
            print(f"  > Target '{target}' not found. Appending to end.")
            final_md += mermaid_block

    return {"final": final_md}

# --- Reducer Subgraph Wiring ---
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_diagrams", decide_diagrams)
reducer_graph.add_node("inject_diagrams", inject_diagrams)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_diagrams")
reducer_graph.add_edge("decide_diagrams", "inject_diagrams")
reducer_graph.add_edge("inject_diagrams", END)

reducer_subgraph = reducer_graph.compile()

# ==========================================
# 8. MAIN GRAPH WIRING
# ==========================================
g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")

# Define exit from reducer
def save_final(state: State):
    final_md = state.get("final", "")
    plan = state.get("plan", {})
    title = plan.get("blog_title", "blog_post")
    
    safe_title = re.sub(r"[^a-zA-Z0-9]", "_", title)
    filename = f"{safe_title}.md"
    try:
        Path(filename).write_text(final_md, encoding="utf-8")
        print(f"SUCCESS: Saved blog to {filename}")
        print("Note: Use a Markdown viewer with Mermaid support to view diagrams.")
    except Exception as e:
        print(f"Error saving file: {e}")

g.add_node("saver", save_final)
g.add_edge("reducer", "saver")
g.add_edge("saver", END)

app = g.compile()


def generate_blog(topic: str):
    initial_state = {
        "topic": topic,
        "sections": [],
        "evidence": [],
        "diagram_specs": []
    }

    final_state = app.invoke(initial_state)

    return {
        "blog_title": final_state["plan"]["blog_title"],
        "audience": final_state["plan"]["audience"],
        "tone": final_state["plan"]["tone"],
        "markdown": final_state["final"]
    }