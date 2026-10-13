import hashlib
import logging
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from llm import (
    generate_category_name,
    infer_document_category,
    find_similar_existing_category_name,
    generate_rag_answer,
    generate_rag_answer_stream,
    generate_conversation_title,
)
from knowledge_graph import KnowledgeGraph
from text_extraction import extract_text, SUPPORTED_EXTENSIONS

_GRAPH_INDEXER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graph-index")
_LOGGER = logging.getLogger(__name__)


class SmartDocEngine:
    """
    Regroupe toute la logique de classification et de RAG pour 
    un utilisateur donné (data_path pointe vers son dossier personnel).
    """

    CACHE_FILENAME = ".embeddings_cache.pkl"
    DOCUMENT_COLUMNS = ["filename", "text", "category", "path", "document_id"]
    CHUNK_COLUMNS = [
        "chunk_id", "document_id", "filename", "category", "chunk_index", "text"
    ]

    def __init__(self, data_path, embedding_model, threshold=0.5,
                 high_confidence_threshold=0.95, chunk_size=100, chunk_overlap=20):

        self.data_path = Path(data_path)
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.df = pd.DataFrame(columns=self.DOCUMENT_COLUMNS)
        self.document_embeddings = None
        self.category_document_embeddings = {}

        self.chunks_df = pd.DataFrame(columns=self.CHUNK_COLUMNS)
        self.chunk_embeddings = None

        self._cache = {}
        self.graph = KnowledgeGraph(self.data_path)

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
        self._sync_graph_documents()

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

            for file_path in sorted(category_path.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    text = extract_text(file_path)
                except Exception:
                    # Fichier corrompu ou illisible : on l'ignore plutôt que
                    # de faire planter le chargement de tout l'espace.
                    continue

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

        self.df = pd.DataFrame(documents, columns=self.DOCUMENT_COLUMNS)
        self.df["document_id"] = self.df["document_id"].astype("object")

        if len(self.df) == 0:
            embedding_dim = self.embedding_model.get_embedding_dimension()
            self.document_embeddings = np.empty((0, embedding_dim))
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

        self.chunks_df = pd.DataFrame(chunk_records, columns=self.CHUNK_COLUMNS)
        self.chunks_df["document_id"] = self.chunks_df["document_id"].astype("object")

        if len(chunk_records) == 0:
            embedding_dim = self.embedding_model.get_embedding_dimension()
            self.chunk_embeddings = np.empty((0, embedding_dim))
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

    def _sync_graph_documents(self):
        """Synchronize persistent graph records and attach stable IDs to rows."""
        document_keys = set()
        pending_graphs = []
        for index, row in self.df.iterrows():
            chunks = self.chunks_df[
                (self.chunks_df["category"] == row["category"])
                & (self.chunks_df["filename"] == row["filename"])
            ]["text"].tolist()
            category = row["category"]
            filename = row["filename"]
            if self.graph.needs_reindex(category, filename, row["text"], chunks):
                document_id = self.graph.sync_document(
                    category,
                    filename,
                    row["text"],
                    chunks,
                    graph_indexed=False
                )
                pending_graphs.append((document_id, row["text"], chunks, category, filename))
            else:
                document_id = self.graph.get_document_id(category, filename)
            self.df.at[index, "document_id"] = document_id
            mask = (
                (self.chunks_df["category"] == category)
                & (self.chunks_df["filename"] == filename)
            )
            self.chunks_df.loc[mask, "document_id"] = document_id
            self.chunks_df.loc[mask, "chunk_id"] = [
                f"{document_id}:{chunk_index}"
                for chunk_index in self.chunks_df.loc[mask, "chunk_index"]
            ]
            document_keys.add((category, filename))

        self.graph.retain_documents(document_keys)
        for document_id, text, chunks, category, filename in pending_graphs:
            self._schedule_graph_index(document_id, text, chunks, category, filename)

    def _schedule_graph_index(self, document_id, text, chunks, category, filename):
        _GRAPH_INDEXER.submit(
            self._index_document_graph,
            document_id,
            text,
            chunks,
            category,
            filename
        )

    def _index_document_graph(self, document_id, text, chunks, category, filename):
        try:
            extracted_graph = self.graph.extract_document(chunks)
            updated = self.graph.replace_document_graph(
                document_id, text, chunks, extracted_graph
            )
            if not updated:
                _LOGGER.info(
                    "Skipped stale graph-index result for %s/%s",
                    category,
                    filename
                )
        except Exception:
            _LOGGER.exception(
                "Graph indexing failed for %s/%s",
                category,
                filename
            )

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
        """
        Retourne la meilleure catégorie par score moyen (top-k), ainsi que la
        meilleure similarité document-à-document trouvée toutes catégories
        confondues (et sa catégorie). Cette dernière sert à détecter les
        quasi-doublons même dans une catégorie qui contient déjà plusieurs
        documents hétérogènes (où la moyenne top-k serait diluée).
        """

        best_category = None
        best_score = -1
        best_document = None

        best_single_score = -1
        best_single_category = None

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

            category_single_score = float(scores[max_index])
            if category_single_score > best_single_score:
                best_single_score = category_single_score
                best_single_category = category

        return best_category, best_score, best_document, best_single_category, best_single_score

    def create_category(self, category_name, document_embedding):

        self.category_document_embeddings[category_name] = [document_embedding]
        (self.data_path / category_name).mkdir(parents=True, exist_ok=True)

    def _remove_existing_document(self, category, filename):
        """
        Supprime (embedding + ligne df + chunks) un document de ce nom déjà
        présent dans cette catégorie précise, s'il existe. Appelé avant de
        ranger un nouveau document au même endroit, pour remplacer proprement
        un réimport à l'identique plutôt que de dupliquer l'entrée.
        """

        mask = (self.df["category"] == category) & (self.df["filename"] == filename)
        if not mask.any():
            return

        category_df = self.df[self.df["category"] == category].reset_index(drop=True)
        position = category_df.index[category_df["filename"] == filename][0]

        if category in self.category_document_embeddings and position < len(self.category_document_embeddings[category]):
            del self.category_document_embeddings[category][position]

        self.df = self.df[~mask].reset_index(drop=True)

        keep_mask = ~((self.chunks_df["category"] == category) & (self.chunks_df["filename"] == filename))
        keep_mask = keep_mask.values
        self.chunks_df = self.chunks_df[keep_mask].reset_index(drop=True)
        self.chunk_embeddings = self.chunk_embeddings[keep_mask]

    def categorize(self, file_path):
        """
        Classe un document et le déplace dans le bon dossier.
        Retourne un dict avec category, score, action, path.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            return {"error": f"Fichier introuvable : {file_path}"}

        try:
            text = extract_text(file_path)
        except Exception as exc:
            return {"error": f"Impossible de lire {file_path.name} : {exc}"}

        if not text.strip():
            return {"error": f"Le document est vide : {file_path.name}"}

        same_document = self.df[
            (self.df["filename"].str.casefold() == file_path.name.casefold())
            & (self.df["text"] == text)
        ]
        if not same_document.empty:
            existing = same_document.iloc[0]
            return {
                "action": "already_classified",
                "category": existing["category"],
                "filename": existing["filename"],
                "path": existing["path"],
            }

        embedding = self.embedding_model.encode([text], normalize_embeddings=True)[0]

        category, score, closest_document, duplicate_category, duplicate_score = (
            self.find_best_category(embedding)
        )

        # Un score quasi parfait avec UN document précis (pas la moyenne de la
        # catégorie, qui serait diluée si la catégorie contient d'autres
        # documents moins similaires) signifie un document identique ou quasi
        # identique à un document déjà classé (ex : le même fichier réimporté).
        # Dans ce cas on fait confiance à l'embedding seul : appeler le LLM
        # pour "confirmer" par un nom de catégorie n'apporte rien et peut au
        # contraire faire échouer un doublon évident, le LLM n'étant pas
        # parfaitement déterministe d'un appel à l'autre.
        very_confident_match = (
            duplicate_category is not None and duplicate_score >= self.high_confidence_threshold
        )
        inferred_category = infer_document_category(text, file_path.name)

        if very_confident_match:
            destination_category = duplicate_category
            action = "existing_category"
        elif inferred_category:
            existing_match = find_similar_existing_category_name(
                inferred_category,
                self.category_document_embeddings.keys()
            )
            destination_category = existing_match or inferred_category
            action = "existing_category" if existing_match else "new_category"
        elif category is not None and score >= max(self.threshold, 0.65):
            destination_category = category
            action = "existing_category"
        else:
            proposed_name = generate_category_name(text)
            existing_match = find_similar_existing_category_name(
                proposed_name,
                self.category_document_embeddings.keys()
            )
            destination_category = existing_match or proposed_name
            action = "existing_category" if existing_match else "new_category"

        new_chunks = self._chunk_text(text)
        destination_folder = self.data_path / destination_category
        destination_folder.mkdir(parents=True, exist_ok=True)
        destination_file = destination_folder / file_path.name
        filename_index = 2
        while destination_file.exists() or (
            (self.df["category"] == destination_category)
            & (self.df["filename"].str.casefold() == destination_file.name.casefold())
        ).any():
            destination_file = destination_folder / (
                f"{file_path.stem} ({filename_index}){file_path.suffix}"
            )
            filename_index += 1
        os.replace(file_path, destination_file)

        if destination_category in self.category_document_embeddings:
            self.category_document_embeddings[destination_category].append(embedding)
        else:
            self.create_category(destination_category, embedding)

        # Met à jour df et les chunks pour que le document soit immédiatement cherchable
        self._register_new_document(destination_file, destination_category, text, new_chunks)

        return {
            "category": destination_category,
            "score": float(score),
            "action": action,
            "filename": destination_file.name,
            "path": str(destination_file)
        }

    def _register_new_document(self, file_path, category, text, new_chunks=None):
        """Ajoute le nouveau document à df et met à jour l'index de chunks."""

        if new_chunks is None:
            new_chunks = self._chunk_text(text)
        document_id = self.graph.sync_document(
            category,
            file_path.name,
            text,
            new_chunks,
            graph_indexed=False
        )
        needs_graph_index = self.graph.needs_reindex(
            category, file_path.name, text, new_chunks
        )
        new_row = {
            "filename": file_path.name,
            "text": text,
            "category": category,
            "path": str(file_path),
            "document_id": document_id
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        chunk_records = []
        for i, chunk in enumerate(new_chunks):
            chunk_records.append({
                "chunk_id": f"{document_id}:{i}",
                "document_id": document_id,
                "filename": file_path.name,
                "category": category,
                "chunk_index": i,
                "text": chunk
            })

        new_chunks_df = pd.DataFrame(chunk_records, columns=self.CHUNK_COLUMNS)
        new_chunk_embeddings = self.embedding_model.encode(
            new_chunks_df["text"].tolist(),
            normalize_embeddings=True
        )

        self.chunks_df = pd.concat([self.chunks_df, new_chunks_df], ignore_index=True)
        self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_chunk_embeddings])
        if needs_graph_index:
            self._schedule_graph_index(
                document_id, text, new_chunks, category, file_path.name
            )

    # --------------------------------------------------------
    # SUPPRESSION
    # --------------------------------------------------------

    def delete_document(self, filename, category=None):
        """Supprime un document (fichier + index en mémoire)."""

        rows = self.df[self.df["filename"] == filename]
        if category is not None:
            rows = rows[rows["category"] == category]
        if rows.empty:
            return {"error": f"Document introuvable : {filename}"}
        if len(rows) > 1:
            return {"error": f"Plusieurs documents portent le nom {filename}; précise le dossier."}

        category = rows.iloc[0]["category"]
        file_path = Path(rows.iloc[0]["path"])
        document_id = rows.iloc[0].get("document_id")

        cat_df = self.df[self.df["category"] == category].reset_index(drop=True)
        position = cat_df.index[cat_df["filename"] == filename][0]

        if category in self.category_document_embeddings:
            del self.category_document_embeddings[category][position]
            if not self.category_document_embeddings[category]:
                del self.category_document_embeddings[category]

        self.df = self.df[
            ~((self.df["filename"] == filename) & (self.df["category"] == category))
        ].reset_index(drop=True)

        keep_mask = (
            self.chunks_df["document_id"] != document_id
        ).values
        self.chunks_df = self.chunks_df[keep_mask].reset_index(drop=True)
        self.chunk_embeddings = self.chunk_embeddings[keep_mask]
        self.graph.remove_document(category, filename)

        if file_path.exists():
            file_path.unlink()
        if file_path.parent.exists() and not any(file_path.parent.iterdir()):
            file_path.parent.rmdir()

        return {"deleted": filename, "category": category}

    def delete_category(self, category):
        """Supprime une catégorie entière (tous ses documents + le dossier)."""

        filenames = self.df[self.df["category"] == category]["filename"].tolist()
        for filename in filenames:
            self.delete_document(filename, category)

        folder = self.data_path / category
        if folder.exists():
            shutil.rmtree(folder)

        return {"deleted_category": category, "count": len(filenames)}

    def rename_category(self, old_name, new_name):
        """Renomme une catégorie : déplace son dossier et met à jour df/chunks/embeddings."""

        old_name = old_name.strip()
        new_name = new_name.strip()

        if not new_name:
            return {"error": "Le nouveau nom ne peut pas être vide."}
        if new_name == old_name:
            return {"error": "Ce nom est déjà celui du dossier."}
        if new_name in self.category_document_embeddings:
            return {"error": f"Une catégorie nommée {new_name} existe déjà."}

        old_folder = self.data_path / old_name
        if not old_folder.exists():
            return {"error": f"Dossier introuvable : {old_name}"}

        new_folder = self.data_path / new_name
        old_folder.rename(new_folder)

        mask = self.df["category"] == old_name
        self.df.loc[mask, "path"] = self.df.loc[mask, "path"].apply(
            lambda p: str(new_folder / Path(p).name)
        )
        self.df.loc[mask, "category"] = new_name

        self.category_document_embeddings[new_name] = self.category_document_embeddings.pop(old_name)

        chunk_mask = self.chunks_df["category"] == old_name
        self.chunks_df.loc[chunk_mask, "category"] = new_name
        self.graph.rename_category(old_name, new_name)

        return {"renamed_category": old_name, "to": new_name}

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
        document_id = rows.iloc[0]["document_id"]
        if not old_path.exists():
            return {"error": f"Fichier introuvable : {old_path}"}

        new_path = self.data_path / new_category / filename
        if new_path.exists():
            return {"error": f"Un document nommé {filename} existe déjà dans {new_category}."}

        try:
            text = extract_text(old_path)
        except Exception as exc:
            return {"error": f"Impossible de lire {filename} : {exc}"}

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
        self.graph.move_document(old_category, filename, new_category)

        # Met à jour les chunks associés (même catégorie, pour rester cohérent avec le RAG)
        chunk_mask = self.chunks_df["document_id"] == document_id
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
        seed_count = min(len(similarities), max(top_k, top_k * 2))
        top_indices = np.argsort(similarities)[::-1][:seed_count]

        results = {}
        seed_chunk_ids = []
        for idx in top_indices:
            row = self.chunks_df.iloc[idx]
            chunk_id = row["chunk_id"]
            seed_chunk_ids.append(chunk_id)
            results[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": row["document_id"],
                "filename": row["filename"],
                "category": row["category"],
                "chunk_text": row["text"],
                "score": float(similarities[idx]),
                "graph_links": []
            }

        for related in self.graph.expand_from_chunks(seed_chunk_ids, limit=top_k * 4):
            graph_score = min(0.75, 0.45 + 0.05 * len(related["matched_entities"]))
            result = results.get(related["chunk_id"])
            if result is None:
                result = {
                    **related,
                    "document_id": related["chunk_id"].rsplit(":", 1)[0],
                    "score": graph_score
                }
                results[related["chunk_id"]] = result
            else:
                result["score"] = max(result["score"], graph_score)
                result["graph_links"] = related["graph_links"]

        return sorted(
            results.values(),
            key=lambda result: (-result["score"], result["category"], result["filename"])
        )[:top_k]

    def _prepare_answer(self, query, top_k, conversation_history):
        conversation_history = conversation_history or []
        previous_questions = [
            message["content"]
            for message in conversation_history
            if message["role"] == "user"
        ][-2:]
        retrieval_query = "\n".join([*previous_questions, query])
        relevant_chunks = self.search_chunks(retrieval_query, top_k=top_k)
        if not relevant_chunks:
            return [], "", []

        context = "\n\n".join([
            f"[Source: {chunk['category']}/{chunk['filename']}]"
            + (
                f" [Lien du graphe: {'; '.join(chunk['graph_links'])}]"
                if chunk.get("graph_links") else ""
            )
            + f"\n{chunk['chunk_text']}"
            for chunk in relevant_chunks
        ])
        sources = list(dict.fromkeys([
            f"{chunk['category']}/{chunk['filename']}" for chunk in relevant_chunks
        ]))
        return relevant_chunks, context, sources

    def answer(self, query, top_k=3, conversation_history=None):
        relevant_chunks, context, sources = self._prepare_answer(
            query, top_k, conversation_history
        )
        if not relevant_chunks:
            return {
                "answer": "Aucun document n'est encore disponible pour répondre à cette question.",
                "sources": [],
                "chunks_used": []
            }

        answer_text = generate_rag_answer(
            query,
            context,
            conversation_history=conversation_history
        )

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": relevant_chunks
        }

    def stream_answer(self, query, top_k=3, conversation_history=None):
        relevant_chunks, context, sources = self._prepare_answer(
            query, top_k, conversation_history
        )
        if not relevant_chunks:
            return {
                "answer_stream": iter([
                    "Aucun document n'est encore disponible pour répondre à cette question."
                ]),
                "sources": [],
                "chunks_used": [],
            }

        return {
            "answer_stream": generate_rag_answer_stream(
                query,
                context,
                conversation_history=conversation_history,
            ),
            "sources": sources,
            "chunks_used": relevant_chunks,
        }

    def suggest_conversation_title(self, query):
        return generate_conversation_title(query)