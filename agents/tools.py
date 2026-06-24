"""Outils personnalisés disponibles pour les agents de l'assistant de recherche."""

from __future__ import annotations

import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from agents.config import get_llm


# =========================
# OUTIL DE RECHERCHE WEB (TAVILY)
# =========================
@tool
def web_search(query: str) -> str:
    """Rechercher sur le web via l'API Tavily."""

    from langchain_tavily import TavilySearch

    if not os.getenv("TAVILY_API_KEY"):
        return "ERREUR : TAVILY_API_KEY manquante dans l'environnement (.env)."

    # TavilySearch lit TAVILY_API_KEY depuis l'env automatiquement
    search = TavilySearch(max_results=5)

    # .invoke() attend un dict {"query": ...}
    raw = search.invoke({"query": query})

    # Normaliser la sortie (Tavily peut retourner dict/list/str)
    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return str(raw)

    if not results:
        return "Aucun résultat trouvé."

    formatted_results = []

    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            title = r.get("title", "Sans titre")
            url = r.get("url", "")
            content = r.get("content", "")
        else:
            title = "Résultat"
            url = ""
            content = str(r)

        formatted_results.append(
            f"[{i}] {title}\n{url}\n{content}"
        )

    return "\n\n".join(formatted_results)


# =========================
# OUTIL DE RÉSUMÉ
# =========================
@tool
def summarize(text: str) -> str:
    """Résumer un texte via le LLM configuré."""

    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(
            content=(
                "Tu es un assistant spécialisé dans la synthèse concise. "
                "Résume le texte en 5 points maximum."
            )
        ),
        HumanMessage(content=text),
    ]

    response = llm.invoke(messages)
    return response.content