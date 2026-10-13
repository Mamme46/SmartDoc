import re
import unicodedata
from difflib import SequenceMatcher

import ollama


def _normalize_category_text(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def infer_document_category(text, filename=""):
    """Return a concise French document-type label when the evidence is clear."""
    normalized = _normalize_category_text(f"{filename}\n{text[:3000]}")
    email_markers = r"\b(courriels?|e[- ]?mails?|emails?|objet)\b"
    if re.search(email_markers, normalized):
        if re.search(r"\b(fournisseur|livraison|commande|achats)\b", normalized):
            return "Courriels fournisseurs"
        return "Correspondance"

    patterns = (
        (r"\b(note de reunion|compte[- ]rendu|proces[- ]verbal|minutes de reunion)\b", "Comptes rendus"),
        (r"\b(note de frais|remboursement de frais|justificatifs de frais)\b", "Notes de frais"),
        (r"\b(procedure|mode operatoire|etapes de validation)\b", "Procédures"),
        (r"\b(fiche fournisseur|presentation du fournisseur)\b", "Fournisseurs"),
        (r"\b(note de service|communication interne)\b", "Notes internes"),
        (r"\b(bulletin de paie|fiche de paie)\b", "Bulletins de paie"),
        (r"\b(releve bancaire|releve de compte)\b", "Relevés bancaires"),
        (r"\b(contrat|convention|avenant)\b", "Contrats"),
        (r"\b(facture|montant ttc|total a payer)\b", "Factures"),
        (r"\b(devis|estimation des couts)\b", "Devis"),
        (r"\b(rapport annuel|rapport d activite|bilan d activite)\b", "Rapports"),
        (r"\b(cv|curriculum vitae|lettre de motivation)\b", "Candidatures"),
    )
    for pattern, category in patterns:
        if re.search(pattern, normalized):
            return category

    recipe_heading = r"\b(recette|ingredients|preparation|etapes de preparation)\b"
    if (
        re.search(recipe_heading, normalized)
        and re.search(r"\b(ingredients|preparation|cuisson|four|minutes)\b", normalized)
    ):
        return "Recettes"

    return None


def generate_category_name(text, max_chars=800):
    """
    Utilise Qwen2.5:3b en local via Ollama pour générer 
    un nom de catégorie court à partir du contenu d'un document.
    """

    excerpt = text[:max_chars]

    prompt = f"""Voici un extrait d'un document :

---
{excerpt}
---

Donne un nom de catégorie court (2 à 3 mots maximum) qui résume 
le sujet principal de ce document, en français.

Règles strictes :
- Réponds UNIQUEMENT avec le nom de la catégorie, rien d'autre
- Pas de phrase, pas d'explication, pas de ponctuation, pas de guillemets
- Utilise une forme générale (ex: "Recettes" et non "Recette de pâtes")
- Si le document correspond à un type administratif courant (facture, contrat, recette, article...), nomme la catégorie d'après ce type plutôt que d'après son sujet précis (ex: "Contrats" et non "Bail" ou "Assurance habitation")

Nom de catégorie :"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 12}
    )

    category_name = response["message"]["content"].strip()
    category_name = category_name.split("\n")[0]
    category_name = category_name.strip('"\'.,: ')
    category_name = category_name.replace("/", "-").replace("\\", "-")

    if len(category_name.split()) > 4:
        category_name = " ".join(category_name.split()[:2])

    return category_name[:1].upper() + category_name[1:60]


def find_similar_existing_category_name(new_name, existing_categories, threshold=0.85):
    """
    Vérifie si un nom très proche existe déjà (évite les doublons
    du type "Voyage" vs "Voyages").
    """

    for existing in existing_categories:
        similarity = SequenceMatcher(None, new_name.lower(), existing.lower()).ratio()
        if similarity >= threshold:
            return existing

    return None


def is_name_compatible(name_a, name_b, threshold=0.4):
    """
    Vérifie si deux noms de catégorie désignent probablement le même type de
    document. Seuil volontairement plus bas que find_similar_existing_category_name :
    sert à confirmer (ou non) une fusion déjà proposée par similarité d'embedding,
    pas à détecter des quasi-doublons de nom.
    """

    if not name_a or not name_b:
        return False

    return SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio() >= threshold


def _build_rag_prompt(query, context, conversation_history):
    history_text = ""
    if conversation_history:
        history_text = "\n\nÉchanges précédents de cette conversation (à utiliser uniquement pour comprendre les références de la question) :\n"
        history_text += "\n".join(
            f"{'Utilisateur' if message['role'] == 'user' else 'Assistant'} : {message['content']}"
            for message in conversation_history[-8:]
        )

    prompt = f"""Tu es l'assistant de SmartDoc, une application qui aide à retrouver 
des informations dans des documents personnels.

Voici des extraits de documents potentiellement pertinents :

{context}
{history_text}

Question de l'utilisateur : {query}

Consignes :
- Réponds UNIQUEMENT à partir des informations présentes dans les extraits ci-dessus
- Si les extraits ne permettent pas de répondre, dis-le clairement
- Après chaque information factuelle, indique la source entre parenthèses
- Réponds en français, de façon claire et concise

Réponse :"""
    return prompt


def generate_rag_answer(query, context, temperature=0.3, conversation_history=None):
    """
    Génère une réponse en langage naturel à partir d'une question
    et d'un contexte de chunks pertinents.
    """
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{
            "role": "user",
            "content": _build_rag_prompt(query, context, conversation_history),
        }],
        options={"temperature": temperature}
    )

    return response["message"]["content"].strip()


def generate_rag_answer_stream(query, context, temperature=0.3, conversation_history=None):
    """Yield response text as Ollama generates it."""
    response_stream = ollama.chat(
        model="qwen2.5:3b",
        messages=[{
            "role": "user",
            "content": _build_rag_prompt(query, context, conversation_history),
        }],
        options={"temperature": temperature},
        stream=True,
    )

    for chunk in response_stream:
        content = chunk["message"]["content"]
        if content:
            yield content


def generate_conversation_title(query):
    prompt = f"""Résume l'idée principale de la question ci-dessous en un titre de conversation court.

Consignes :
- Réponds uniquement par un titre en français, sans guillemets ni ponctuation finale
- Reformule le sujet, ne recopie pas la question
- Utilise 3 à 6 mots, sans ajouter d'information absente de la question

Question : {query}

Titre :"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_predict": 24}
    )

    title = response["message"]["content"].strip().strip("\"'“”‘’` ")
    title = " ".join(title.split()).rstrip(" .!?;:")
    if not title:
        raise ValueError("Le modèle n'a pas renvoyé de titre de conversation.")
    return title[:60].rstrip()