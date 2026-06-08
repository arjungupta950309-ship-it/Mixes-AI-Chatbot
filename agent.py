"""Self-correcting RAG agent for King Arthur Baking mixes, built with LangGraph.

This is an *agentic* retrieval pipeline, not a single retrieve-then-answer
chain. The graph reasons about whether the documents it pulled are actually
relevant and, if not, rewrites the query and tries again before answering:

    retrieve -> grade -> (relevant?) -- yes --> generate -> END
                              |
                              no, and retries left
                              v
                          rewrite -> retrieve (loop)

Grading and rewriting give the agent "advanced knowledge retrieval and
reasoning": it self-checks retrieval quality and reformulates poor queries.

The compiled graph is exposed via build_graph() so the frontend can render
its structure (graph.get_graph().draw_mermaid_png()).
"""

from __future__ import annotations

import re
from typing import Annotated, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import config
from providers import get_embeddings, get_llm

MAX_REWRITES = 2
RETRIEVE_K = 6


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]  # full chat history
    question: str            # the (possibly rewritten) query used for retrieval
    original_question: str   # what the user actually asked
    documents: list[Document]
    relevant: bool           # grader verdict on the current documents
    generation: str
    rewrites: int


# --------------------------------------------------------------------------- #
# Lazily-constructed singletons (built once, reused across turns)
# --------------------------------------------------------------------------- #
_vectorstore: Chroma | None = None
_gen_llm = None
_short_llm = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def gen_llm():
    """LLM for writing the final answer (longer output budget)."""
    global _gen_llm
    if _gen_llm is None:
        _gen_llm = get_llm(temperature=0.2)
    return _gen_llm


def short_llm():
    """LLM for grade/rewrite — greedy and tiny output, so it's fast on CPU."""
    global _short_llm
    if _short_llm is None:
        _short_llm = get_llm(temperature=0.0, max_new_tokens=24)
    return _short_llm


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def retrieve(state: AgentState) -> dict:
    """Semantic search over the product knowledge base."""
    docs = get_vectorstore().similarity_search(state["question"], k=RETRIEVE_K)
    return {"documents": docs}


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether a set of retrieved baking-mix products is "
            "ON-TOPIC for the user's question. Judge the TOPIC, not whether the "
            "answer is spelled out.\n"
            "- Answer YES if the products are the kind of thing the user asked "
            "about (e.g. cookie mixes for a question about cookies, or any mixes "
            "for a question about prices/ratings). For 'cheapest', 'best', or "
            "'list' questions, relevant products are always YES.\n"
            "- Answer NO only if the products are about something COMPLETELY "
            "different from the question.\n"
            "Reply with exactly one word: YES or NO.",
        ),
        ("human", "Question: {q}\n\nProducts:\n{ctx}\n\nOn-topic? YES or NO."),
    ]
)


def grade_documents(state: AgentState) -> dict:
    """Ask the LLM whether the retrieved products are relevant to the question.

    Reasoning step: if retrieval missed, we'd rather rewrite than answer from
    irrelevant context. Uses a simple YES/NO completion so it works across all
    providers (no tool-calling required).
    """
    if not state["documents"]:
        return {"relevant": False}
    context = "\n\n".join(d.page_content for d in state["documents"])
    try:
        resp = (GRADE_PROMPT | short_llm()).invoke(
            {"q": state["original_question"], "ctx": context}
        )
        text = resp.content if isinstance(resp, AIMessage) else str(resp)
        # Lenient parse: only treat as irrelevant when the FIRST word is "no".
        # (Avoids "not", "none", etc. being misread as a rejection.)
        words = re.findall(r"[a-z]+", text.lower())
        relevant = not (words and words[0] == "no")
    except Exception:
        # If grading fails for any reason, don't block the answer.
        relevant = True
    # NOTE: never discard the documents — grading only decides whether to try a
    # better search. Generation always keeps the retrieved products as context.
    return {"relevant": relevant}


def rewrite_query(state: AgentState) -> dict:
    """Reformulate the question to improve retrieval."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's question into a better search query for a "
                "catalog of baking mixes. Expand abbreviations, add synonyms for "
                "key concepts, and focus on product attributes (flavor, type, "
                "dietary needs). Return only the rewritten query.",
            ),
            ("human", "{q}"),
        ]
    )
    new_q = (prompt | short_llm()).invoke({"q": state["original_question"]})
    text = new_q.content if isinstance(new_q, AIMessage) else str(new_q)
    return {"question": text.strip(), "rewrites": state["rewrites"] + 1}


GENERATE_SYSTEM = """You are the King Arthur Baking mixes assistant. Answer the \
user's question using ONLY the product information in the context below. The \
context is the catalog's "Mixes" category.

Guidelines:
- Keep the answer SHORT: at most 3-4 sentences or a short bulleted list. Do not \
repeat the question or pad the answer.
- Recommend specific products by name with price and dietary attributes (e.g. \
gluten-free, kosher) when relevant.
- For "cheapest/most expensive" questions, COMPARE the Price values in the context \
and pick the correct one. For "best/highest rated" or "most reviews" questions, \
COMPARE the Rating and review counts. Read the numbers carefully.
- If the user asks about dietary needs, only claim an attribute if the product's \
listed attributes confirm it.
- If the context does not contain the answer, say you couldn't find a matching \
mix rather than inventing one.

Context:
{context}"""


def generate(state: AgentState) -> dict:
    """Produce the grounded final answer."""
    if state["documents"]:
        context = "\n\n".join(d.page_content for d in state["documents"])
    else:
        context = "(no matching products were found in the catalog)"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATE_SYSTEM),
            *[(m.type, m.content) for m in state["messages"][:-1]],  # prior turns
            ("human", "{question}"),
        ]
    )
    answer = (prompt | gen_llm()).invoke(
        {"context": context, "question": state["original_question"]}
    )
    text = answer.content if isinstance(answer, AIMessage) else str(answer)
    return {"generation": text, "messages": [AIMessage(content=text)]}


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #
def decide_after_grading(state: AgentState) -> str:
    """Decide whether to answer now or try a better search.

    We answer when the documents are judged relevant OR we've exhausted our
    rewrite budget. Crucially, the retrieved documents are always kept, so a
    misfiring grader can only cost an extra search — never an empty answer.
    """
    if state.get("relevant", True):
        return "generate"
    if state["rewrites"] >= MAX_REWRITES:
        return "generate"  # out of retries; answer from what we have
    return "rewrite"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    """Compile and return the LangGraph agent.

    Default (smart) graph self-corrects: retrieve -> grade -> rewrite/generate.
    With AGENT_FAST=1 it collapses to retrieve -> generate for lower latency.
    """
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")

    if config.AGENT_FAST:
        g.add_edge("retrieve", "generate")
    else:
        g.add_node("grade", grade_documents)
        g.add_node("rewrite", rewrite_query)
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            decide_after_grading,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        g.add_edge("rewrite", "retrieve")

    g.add_edge("generate", END)
    return g.compile()


def ask(graph, question: str, history: list[BaseMessage] | None = None) -> dict:
    """Convenience wrapper: run one turn and return {answer, sources}."""
    history = history or []
    state = {
        "messages": history + [HumanMessage(content=question)],
        "question": question,
        "original_question": question,
        "documents": [],
        "relevant": True,
        "generation": "",
        "rewrites": 0,
    }
    result = graph.invoke(state)
    sources = [
        {"title": d.metadata.get("title", ""),
         "url": d.metadata.get("url", ""),
         "price": d.metadata.get("price", ""),
         "image": d.metadata.get("image", ""),
         "rating": d.metadata.get("rating", 0),
         "review_count": d.metadata.get("review_count", 0),
         "badges": d.metadata.get("badges", "")}
        for d in result.get("documents", [])
    ]
    return {"answer": result["generation"], "sources": sources}


if __name__ == "__main__":
    # Quick smoke test from the command line.
    graph = build_graph()
    out = ask(graph, "What gluten-free cookie mixes do you have?")
    print("\nANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(" -", s["title"], s["price"], s["url"])
