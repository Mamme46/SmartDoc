import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model():
    """
    Charge le modèle d'embeddings une seule fois (singleton),
    même si cette fonction est appelée plusieurs fois dans l'app.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _model