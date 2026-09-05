import shutil
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from llm import (
    generate_category_name,
    find_similar_existing_category_name,
    is_name_compatible,
    generate_rag_answer,
)


class SmartDocEngine:
    """
    Regroupe toute la logique de classification et de RAG pour 
    un utilisateur donné (data_path pointe vers son dossier personnel).
    """

    CACHE_FILENAME = ".embeddings_cache.pkl"

    def __init__(self, data_path, embedding_model, threshold=0.5,
                 chunk_size=100, chunk_overlap=20):

        self.data_path = Path(data_path)
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.df = pd.DataFrame()
        self.document_embeddings = None
        self.category_document_embeddings = {}

        self.chunks_df = pd.DataFrame()
        self.chunk_embeddings = None

        self._cache = {}

        self.load()

    # --------------------------------------------------------
    # CHARGEMENT INITIAL
    # --------------------------------------------------------

    def load(self):
        """
        Charge tous les documents existants et construit les index.

        Réutilise un cache d'embeddings sur disque (par utilisateur) pour
        les documents inchangés depuis le dernier chargement, identifiés par
        empreinte de leur contenu : seuls les documents nouveaux ou modifiés
        sont réellement passés dans le modèle d'embeddings.
        """

        self.data_path.mkdir(parents=True, exist_ok=True)

        old_cache = self._read_cache()
        new_cache = {}

        self._load_documents(old_cache, new_cache)
        self._build_category_embeddings()
        self._build_chunks(old_cache, new_cache)

        self._cache = new_cache
        self._write_cache()

    def _cache_path(self):
        return self.data_path / self.CACHE_FILENAME

    def _read_cache(self):
        path = self._cache_path()
        if not path.exists():
            return {}
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}

    def _write_cache(self):
        with open(self._cache_path(), "wb") as f:
            pickle.dump(self._cache, f)

    @staticmethod
    def _hash_text(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load_documents(self, old_cache, new_cache):

        documents = []
        pending_texts = []
        pending_indices = []
        cached_embeddings = {}

        for category_path in self.data_path.iterdir():
            if not category_path.is_dir():
                continue

            category = category_path.name

            for file_path in category_path.glob("*.txt"):
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                key = f"{category}/{file_path.name}"
                text_hash = self._hash_text(text)

                index = len(documents)
                documents.append({
                    "filename": file_path.name,
                    "text": text,
                    "category": category,
                    "path": str(file_path)
                })

                cached = old_cache.get(key)
                if cached is not None and cached.get("hash") == text_hash:
                    cached_embeddings[index] = cached["embedding"]
                    new_cache[key] = {"hash": text_hash, "embedding": cached["embedding"]}
                else:
                    pending_texts.append(text)
                    pending_indices.append(index)

        self.df = pd.DataFrame(documents)

        if len(self.df) == 0:
            self.document_embeddings = np.empty((0, 768))
            return

        embedding_dim = self.embedding_model.get_embedding_dimension()
        embeddings = np.empty((len(documents), embedding_dim))

        for index, embedding in cached_embeddings.items():
            embeddings[index] = embedding

        if pending_texts:
            fresh = self.embedding_model.encode(pending_texts, normalize_embeddings=True)
            for index, embedding in zip(pending_indices, fresh):
                embeddings[index] = embedding
                key = f"{documents[index]['category']}/{documents[index]['filename']}"
                new_cache[key] = {"hash": self._hash_text(documents[index]["text"]), "embedding": embedding}

        self.document_embeddings = embeddings

    def _build_category_embeddings(self):

        self.category_document_embeddings = {}

        if len(self.df) == 0:
            return

        for category in self.df["category"].unique():
            mask = self.df["category"].values == category
            self.category_document_embeddings[category] = list(self.document_embeddings[mask])

    def _build_chunks(self, old_cache, new_cache):

        chunk_records = []
        doc_blocks = []  # (start, chunk_texts, key, text_hash, embeddings_en_cache_ou_None)

        for _, row in self.df.iterrows():

            key = f"{row['category']}/{row['filename']}"
            text_hash = self._hash_text(row["text"])
            cached = old_cache.get(key)

            if cached is not None and cached.get("hash") == text_hash and "chunk_texts" in cached:
                chunk_texts = cached["chunk_texts"]
                cached_chunk_embeddings = cached["chunk_embeddings"]
            else:
                chunk_texts = self._chunk_text(row["text"])
                cached_chunk_embeddings = None

            start = len(chunk_records)
            for i, chunk in enumerate(chunk_texts):
                chunk_records.append({
                    "chunk_id": f"{row['filename']}_{i}",
                    "filename": row["filename"],
                    "category": row["category"],
                    "chunk_index": i,
                    "text": chunk
                })

            doc_blocks.append((start, chunk_texts, key, text_hash, cached_chunk_embeddings))

        self.chunks_df = pd.DataFrame(chunk_records)

        if len(chunk_records) == 0:
            self.chunk_embeddings = np.empty((0, 768))
            return

        embedding_dim = self.embedding_model.get_embedding_dimension()
        embeddings = np.empty((len(chunk_records), embedding_dim))

        pending_texts = []
        pending_ranges = []

        for start, chunk_texts, key, text_hash, cached_chunk_embeddings in doc_blocks:
            length = len(chunk_texts)
            entry = new_cache.setdefault(key, {"hash": text_hash})

            if length == 0:
                entry["chunk_texts"] = []
                entry["chunk_embeddings"] = np.empty((0, embedding_dim))
            elif cached_chunk_embeddings is not None:
                embeddings[start:start + length] = cached_chunk_embeddings
                entry["chunk_texts"] = chunk_texts
                entry["chunk_embeddings"] = cached_chunk_embeddings
            else:
                pending_texts.extend(chunk_texts)
                pending_ranges.append((start, length, key, chunk_texts))

        if pending_texts:
            fresh = self.embedding_model.encode(pending_texts, normalize_embeddings=True)
            offset = 0
            for start, length, key, chunk_texts in pending_ranges:
                block = fresh[offset:offset + length]
                embeddings[start:start + length] = block
                new_cache[key]["chunk_texts"] = chunk_texts
                new_cache[key]["chunk_embeddings"] = block
                offset += length

        self.chunk_embeddings = embeddings

    def _chunk_text(self, text):

        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    def find_best_category(self, document_embedding, top_k=3):

        best_category = None
        best_score = -1
        best_document = None

        for category, embeddings_list in self.category_document_embeddings.items():

            category_array = np.atleast_2d(np.array(embeddings_list))

            scores = cosine_similarity(
                document_embedding.reshape(1, -1),
                category_array
            )[0]

            k = min(top_k, len(scores))
            top_scores = np.sort(scores)[-k:]
            max_score = float(np.mean(top_scores))
            max_index = int(np.argmax(scores))

            if max_score > best_score:
                best_score = max_score
                best_category = category

                category_df = self.df[self.df["category"] == category].reset_index(drop=True)
                if max_index < len(category_df):
                    best_document = category_df.iloc[max_index]["filename"]
                else:
                    best_document = "document ajouté dynamiquement"

        return best_category, best_score, best_document

    def create_category(self, category_name, document_embedding):

        self.category_document_embeddings[category_name] = [document_embedding]
        (self.data_path / category_name).mkdir(parents=True, exist_ok=True)

    def categorize(self, file_path):
        """
        Classe un document et le déplace dans le bon dossier.
        Retourne un dict avec category, score, action, path.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            return {"error": f"Fichier introuvable : {file_path}"}

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        if not text.strip():
            return {"error": f"Le document est vide : {file_path.name}"}

        embedding = self.embedding_model.encode([text], normalize_embeddings=True)[0]

        category, score, closest_document = self.find_best_category(embedding)

        # Un score élevé ne suffit pas : des documents de nature différente
        # (facture vs contrat, bulletin de paie vs relevé bancaire...) peuvent
        # être proches en embedding à cause de leur mise en forme commune.
        # On confirme donc la fusion avec le nom que le LLM donnerait au
        # document, comparé au nom de la catégorie candidate.
        proposed_name = generate_category_name(text)

        confident_match = (
            score >= self.threshold
            and category is not None
            and is_name_compatible(proposed_name, category)
        )

        if confident_match:
            destination_category = category
            self.category_document_embeddings[category].append(embedding)
            action = "existing_category"

        else:
            existing_match = find_similar_existing_category_name(
                proposed_name,
                self.category_document_embeddings.keys()
            )

            if existing_match:
                destination_category = existing_match
                self.category_document_embeddings[existing_match].append(embedding)
                action = "existing_category"
            else:
                destination_category = proposed_name
                self.create_category(destination_category, embedding)
                action = "new_category"

        destination_folder = self.data_path / destination_category
        destination_folder.mkdir(parents=True, exist_ok=True)
        destination_file = destination_folder / file_path.name
        file_path.rename(destination_file)

        # Met à jour df et les chunks pour que le document soit immédiatement cherchable
        self._register_new_document(destination_file, destination_category, text)

        return {
            "category": destination_category,
            "score": float(score),
            "action": action,
            "path": str(destination_file)
        }

    def _register_new_document(self, file_path, category, text):
        """Ajoute le nouveau document à df et met à jour l'index de chunks."""

        new_row = {
            "filename": file_path.name,
            "text": text,
            "category": category,
            "path": str(file_path)
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        new_chunks = self._chunk_text(text)
        chunk_records = []
        for i, chunk in enumerate(new_chunks):
            chunk_records.append({
                "chunk_id": f"{file_path.name}_{i}",
                "filename": file_path.name,
                "category": category,
                "chunk_index": i,
                "text": chunk
            })

        new_chunks_df = pd.DataFrame(chunk_records)
        new_chunk_embeddings = self.embedding_model.encode(
            new_chunks_df["text"].tolist(),
            normalize_embeddings=True
        )

        self.chunks_df = pd.concat([self.chunks_df, new_chunks_df], ignore_index=True)
        self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_chunk_embeddings])

    # --------------------------------------------------------
    # SUPPRESSION
    # --------------------------------------------------------

    def delete_document(self, filename):
        """Supprime un document (fichier + index en mémoire)."""

        rows = self.df[self.df["filename"] == filename]
        if rows.empty:
            return {"error": f"Document introuvable : {filename}"}

        category = rows.iloc[0]["category"]
        file_path = Path(rows.iloc[0]["path"])

        cat_df = self.df[self.df["category"] == category].reset_index(drop=True)
        position = cat_df.index[cat_df["filename"] == filename][0]

        if category in self.category_document_embeddings:
            del self.category_document_embeddings[category][position]
            if not self.category_document_embeddings[category]:
                del self.category_document_embeddings[category]

        self.df = self.df[self.df["filename"] != filename].reset_index(drop=True)

        keep_mask = (self.chunks_df["filename"] != filename).values
        self.chunks_df = self.chunks_df[keep_mask].reset_index(drop=True)
        self.chunk_embeddings = self.chunk_embeddings[keep_mask]

        if file_path.exists():
            file_path.unlink()
        if file_path.parent.exists() and not any(file_path.parent.iterdir()):
            file_path.parent.rmdir()

        return {"deleted": filename, "category": category}

    def delete_category(self, category):
        """Supprime une catégorie entière (tous ses documents + le dossier)."""

        filenames = self.df[self.df["category"] == category]["filename"].tolist()
        for filename in filenames:
            self.delete_document(filename)

        folder = self.data_path / category
        if folder.exists():
            shutil.rmtree(folder)

        return {"deleted_category": category, "count": len(filenames)}

    # --------------------------------------------------------
    # CORRECTION MANUELLE
    # --------------------------------------------------------

    def correct_classification(self, filename, old_category, new_category):
        """
        Déplace un document mal classé de old_category vers new_category
        (créée si elle n'existe pas encore). Recalcule son embedding et met
        à jour df / chunks / fichier sur disque en conséquence.
        """

        rows = self.df[(self.df["filename"] == filename) & (self.df["category"] == old_category)]
        if rows.empty:
            return {"error": f"Document introuvable dans {old_category} : {filename}"}

        if old_category == new_category:
            return {"error": "La nouvelle catégorie est identique à l'ancienne."}

        old_path = Path(rows.iloc[0]["path"])
        if not old_path.exists():
            return {"error": f"Fichier introuvable : {old_path}"}

        new_path = self.data_path / new_category / filename
        if new_path.exists():
            return {"error": f"Un document nommé {filename} existe déjà dans {new_category}."}

        text = old_path.read_text(encoding="utf-8", errors="ignore")
        embedding = self.embedding_model.encode([text], normalize_embeddings=True)[0]

        # Retire l'embedding de l'ancienne catégorie
        cat_df = self.df[self.df["category"] == old_category].reset_index(drop=True)
        position = cat_df.index[cat_df["filename"] == filename][0]
        if old_category in self.category_document_embeddings:
            del self.category_document_embeddings[old_category][position]
            if not self.category_document_embeddings[old_category]:
                del self.category_document_embeddings[old_category]

        # L'ajoute à la nouvelle catégorie (la crée si besoin)
        if new_category in self.category_document_embeddings:
            self.category_document_embeddings[new_category].append(embedding)
        else:
            self.create_category(new_category, embedding)

        # Déplace le fichier physiquement
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        if old_path.parent.exists() and not any(old_path.parent.iterdir()):
            old_path.parent.rmdir()

        # Met à jour le document dans df
        df_index = self.df.index[
            (self.df["filename"] == filename) & (self.df["category"] == old_category)
        ][0]
        self.df.at[df_index, "category"] = new_category
        self.df.at[df_index, "path"] = str(new_path)
        self.df.at[df_index, "text"] = text

        # Met à jour les chunks associés (même catégorie, pour rester cohérent avec le RAG)
        chunk_mask = self.chunks_df["filename"] == filename
        self.chunks_df.loc[chunk_mask, "category"] = new_category

        return {"corrected": filename, "old_category": old_category, "new_category": new_category}

    # --------------------------------------------------------
    # RAG / CHATBOT
    # --------------------------------------------------------

    def search_chunks(self, query, top_k=3):

        if len(self.chunks_df) == 0:
            return []

        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "filename": self.chunks_df.iloc[idx]["filename"],
                "category": self.chunks_df.iloc[idx]["category"],
                "chunk_text": self.chunks_df.iloc[idx]["text"],
                "score": float(similarities[idx])
            })

        return results

    def answer(self, query, top_k=3):

        relevant_chunks = self.search_chunks(query, top_k=top_k)

        if not relevant_chunks:
            return {
                "answer": "Aucun document n'est encore disponible pour répondre à cette question.",
                "sources": [],
                "chunks_used": []
            }

        context = "\n\n".join([
            f"[Source: {c['filename']}]\n{c['chunk_text']}"
            for c in relevant_chunks
        ])

        answer_text = generate_rag_answer(query, context)

        sources = list(dict.fromkeys([c["filename"] for c in relevant_chunks]))

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": relevant_chunks
        }