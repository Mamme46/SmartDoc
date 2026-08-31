import ollama
from difflib import SequenceMatcher


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
        options={"temperature": 0.2}
    )

    category_name = response["message"]["content"].strip()
    category_name = category_name.split("\n")[0]
    category_name = category_name.strip('"\'.,: ')
    category_name = category_name.replace("/", "-").replace("\\", "-")

    if len(category_name.split()) > 4:
        category_name = " ".join(category_name.split()[:2])

    return category_name.capitalize()


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


def generate_rag_answer(query, context, temperature=0.3):
    """
    Génère une réponse en langage naturel à partir d'une question
    et d'un contexte de chunks pertinents.
    """

    prompt = f"""Tu es l'assistant de SmartDoc, une application qui aide à retrouver 
des informations dans des documents personnels.

Voici des extraits de documents potentiellement pertinents :

{context}

Question de l'utilisateur : {query}

Consignes :
- Réponds UNIQUEMENT à partir des informations présentes dans les extraits ci-dessus
- Si les extraits ne permettent pas de répondre, dis-le clairement
- Après chaque information factuelle, indique la source entre parenthèses
- Réponds en français, de façon claire et concise

Réponse :"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )

    return response["message"]["content"].strip()