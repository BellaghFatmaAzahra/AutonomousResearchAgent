"""Configuration des fournisseurs LLM avec support multi-fournisseurs."""

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",  # corrigé
        "env_key": "ANTHROPIC_API_KEY",
    },
    "ollama": {
        "model": "llama3",
        "env_key": None,  # modèle local → pas de clé API
    },
}


def get_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
):
    """Retourner une instance de modèle de chat selon le fournisseur."""

    provider = (
        provider
        or os.getenv("LLM_PROVIDER", "ollama")
    ).lower().strip()

    if provider not in _DEFAULTS:
        raise ValueError(
            f"Fournisseur inconnu '{provider}'. Choisir parmi : {', '.join(_DEFAULTS)}"
        )

    cfg = _DEFAULTS[provider]
    model = model or cfg["model"]

    # =========================
    # OLLAMA (LOCAL - SANS CLÉ)
    # =========================
    if provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    # =========================
    # FOURNISSEURS CLOUD
    # =========================
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"Clé API manquante : {cfg['env_key']} pour le fournisseur '{provider}'"
        )

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    # repli de sécurité (ne devrait jamais arriver)
    raise ValueError(f"Fournisseur non supporté : {provider}")