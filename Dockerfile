FROM python:3.12-slim AS base

# Installer uv pour une gestion rapide des dépendances
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copier la spécification des dépendances en premier pour un meilleur cache des couches
COPY pyproject.toml ./

# Installer les dépendances de production
RUN uv sync --no-dev --no-install-project

# Copier le code de l'application
COPY . .

# Installer le projet lui-même
RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "python", "main.py"]