FROM python:3.12-slim

WORKDIR /app

# Évite les fichiers .pyc et permet de voir les logs immédiatement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances
COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Code de l'application
COPY app/ ./app/

# Code Python IA utilisé par l'application
COPY ai/src/ ./ai/src/

# Base SQLite
COPY smartdoc.db ./smartdoc.db

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]