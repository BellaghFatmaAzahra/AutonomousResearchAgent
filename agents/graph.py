"""Définition du graphe d'états LangGraph pour l'assistant de recherche multi-agent.

Architecture
------------
Un nœud **Superviseur** inspecte l'état courant et dirige le travail vers l'un
des trois agents spécialistes :

* **Chercheur** -- collecte des informations via la recherche web et la synthèse.
* **Rédacteur** -- transforme les notes de recherche en un brouillon soigné.
* **Réviseur** -- critique le brouillon ; décide *réviser* ou *accepter*.

Des arêtes conditionnelles renvoient le verdict du réviseur au superviseur afin
que la boucle puisse itérer jusqu'à ce que le résultat soit accepté ou qu'un
nombre maximum d'itérations soit atteint.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.config import get_llm
from agents.tools import summarize, web_search

# ---------------------------------------------------------------------------
# État
# ---------------------------------------------------------------------------

MAX_REVISIONS = 3


class AgentState(BaseModel):
    """État partagé circulant dans le graphe."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    research_notes: str = ""
    draft: str = ""
    review_feedback: str = ""
    next_agent: str = "researcher"
    revision_count: int = 0


# ---------------------------------------------------------------------------
# Nœuds
# ---------------------------------------------------------------------------


def supervisor_node(state: AgentState) -> dict:
    """Décider quel agent doit agir ensuite en fonction de l'état courant."""
    llm = get_llm()

    system = SystemMessage(
        content=(
            "Tu es le superviseur d'une équipe de recherche. En fonction de "
            "la conversation jusqu'ici, décide quel agent doit agir ensuite.\n\n"
            "Agents :\n"
            "  - researcher : collecte des informations sur le web\n"
            "  - writer : rédige un rapport à partir des notes de recherche\n"
            "  - reviewer : critique le brouillon\n"
            "  - FINISH : la tâche est terminée\n\n"
            "Règles :\n"
            "  1. S'il n'y a pas encore de notes de recherche, choisir 'researcher'.\n"
            "  2. S'il y a des notes mais pas de brouillon, choisir 'writer'.\n"
            "  3. S'il y a un brouillon mais pas de révision, choisir 'reviewer'.\n"
            "  4. Si la révision dit 'ACCEPT', choisir 'FINISH'.\n"
            "  5. Si la révision dit 'REVISE', choisir 'writer'.\n\n"
            "Répondre UNIQUEMENT avec le nom de l'agent (un seul mot)."
        )
    )

    context_parts: list[str] = []
    if state.research_notes:
        context_parts.append(f"Notes de recherche :\n{state.research_notes[:2000]}")
    if state.draft:
        context_parts.append(f"Brouillon actuel :\n{state.draft[:2000]}")
    if state.review_feedback:
        context_parts.append(f"Retour de révision :\n{state.review_feedback}")

    human = HumanMessage(
        content=(
            "État actuel :\n"
            + ("\n---\n".join(context_parts) if context_parts else "(vide -- aucun travail effectué)")
        )
    )

    response = llm.invoke([system, human])
    next_agent = response.content.strip().lower().replace("'", "").replace('"', "")

    # Normaliser les réponses courantes du LLM
    if "finish" in next_agent:
        next_agent = "FINISH"
    elif "research" in next_agent:
        next_agent = "researcher"
    elif "writ" in next_agent:
        next_agent = "writer"
    elif "review" in next_agent:
        next_agent = "reviewer"

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Superviseur] Routage vers : {next_agent}")],
    }


def researcher_node(state: AgentState) -> dict:
    """Utiliser les outils pour rechercher la requête de l'utilisateur."""
    query = state.messages[0].content if state.messages else "recherche générale"

    search_results = web_search.invoke({"query": query})
    summary = summarize.invoke({"text": search_results})

    return {
        "research_notes": summary,
        "messages": [AIMessage(content=f"[Chercheur] Notes collectées :\n{summary}")],
    }


def writer_node(state: AgentState) -> dict:
    """Produire ou réviser un brouillon à partir des notes de recherche."""
    llm = get_llm()

    revision_context = ""
    if state.review_feedback:
        revision_context = (
            f"\n\nLe réviseur a fourni ce retour sur votre brouillon précédent -- "
            f"répondez à chaque point :\n{state.review_feedback}"
        )

    system = SystemMessage(
        content=(
            "Tu es un rédacteur technique expérimenté. En utilisant les notes de recherche "
            "fournies, rédige un rapport clair et bien structuré (3 à 5 paragraphes). "
            "Utilise le format markdown."
            + revision_context
        )
    )
    human = HumanMessage(
        content=f"Notes de recherche :\n{state.research_notes}\n\nBrouillon précédent :\n{state.draft}"
    )

    response = llm.invoke([system, human])
    draft = response.content

    return {
        "draft": draft,
        "messages": [AIMessage(content=f"[Rédacteur] Brouillon produit ({len(draft)} caractères)")],
    }


def reviewer_node(state: AgentState) -> dict:
    """Critiquer le brouillon et décider de l'accepter ou de demander une révision."""
    llm = get_llm()

    system = SystemMessage(
        content=(
            "Tu es un éditeur minutieux. Révise le brouillon ci-dessous pour son exactitude, "
            "sa clarté et son exhaustivité. Fournis un retour bref et concret.\n\n"
            "Termine ta révision avec exactement l'un de ces verdicts sur sa propre ligne :\n"
            "  ACCEPT -- le brouillon est prêt pour la publication.\n"
            "  REVISE -- le brouillon nécessite des améliorations."
        )
    )
    human = HumanMessage(content=f"Brouillon :\n{state.draft}")

    response = llm.invoke([system, human])
    feedback = response.content

    verdict = "REVISE"
    if "ACCEPT" in feedback.upper().split("\n")[-1]:
        verdict = "ACCEPT"

    # Appliquer le nombre maximum de révisions
    revision_count = state.revision_count + 1
    if revision_count >= MAX_REVISIONS:
        verdict = "ACCEPT"
        feedback += "\n\n[Accepté automatiquement après le nombre maximum de révisions atteint.]"

    return {
        "review_feedback": feedback,
        "revision_count": revision_count,
        "messages": [AIMessage(content=f"[Réviseur] Verdict : {verdict}\n{feedback}")],
    }


# ---------------------------------------------------------------------------
# Helpers de routage
# ---------------------------------------------------------------------------


def route_supervisor(state: AgentState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    """Retourner le nom du prochain nœud selon la décision du superviseur."""
    agent = state.next_agent
    if agent == "FINISH":
        return END
    if agent in {"researcher", "writer", "reviewer"}:
        return agent
    # Repli -- ne devrait pas arriver mais sécurise le graphe.
    return END


def route_after_review(state: AgentState) -> Literal["supervisor"]:
    """Toujours retourner au superviseur après une révision."""
    return "supervisor"


# ---------------------------------------------------------------------------
# Assemblage du graphe
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construire et compiler le graphe de recherche multi-agent."""
    graph = StateGraph(AgentState)

    # Ajouter les nœuds
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Point d'entrée
    graph.set_entry_point("supervisor")

    # Le superviseur route conditionnellement
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END,
        },
    )

    # Après chaque spécialiste, retour au superviseur
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()