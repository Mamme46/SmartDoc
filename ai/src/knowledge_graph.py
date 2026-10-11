import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

from graph_extraction import extract_document_graph


class KnowledgeGraph:
    """Per-user SQLite store for document chunks, entities, and relations."""

    DATABASE_FILENAME = ".knowledge_graph.sqlite3"

    def __init__(self, data_path):
        self.database_path = Path(data_path) / self.DATABASE_FILENAME
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunks_hash TEXT NOT NULL,
                    graph_indexed INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(category, filename)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    UNIQUE(normalized_name, entity_type)
                );
                CREATE TABLE IF NOT EXISTS mentions (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    PRIMARY KEY(chunk_id, entity_id)
                );
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    UNIQUE(source_entity_id, target_entity_id, relation_type, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);
                CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);
            """)
            document_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)")
            }
            if "graph_indexed" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN graph_indexed INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def _hash(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunks_hash(chunks):
        return KnowledgeGraph._hash(json.dumps(chunks, ensure_ascii=False))

    def needs_reindex(self, category, filename, text, chunks):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash, chunks_hash, graph_indexed FROM documents "
                "WHERE category = ? AND filename = ?",
                (category, filename)
            ).fetchone()
        return (
            row is None
            or row["content_hash"] != self._hash(text)
            or row["chunks_hash"] != self._chunks_hash(chunks)
            or not row["graph_indexed"]
        )

    @staticmethod
    def extract_document(chunks):
        return extract_document_graph(chunks)

    def sync_document(self, category, filename, text, chunks, extracted_chunks=None,
                      graph_indexed=True):
        """Create or update graph data, retaining the document identity on edits."""
        content_hash = self._hash(text)
        chunks_hash = self._chunks_hash(chunks)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, content_hash, chunks_hash, graph_indexed FROM documents "
                "WHERE category = ? AND filename = ?",
                (category, filename)
            ).fetchone()
            if (
                existing is not None
                and existing["content_hash"] == content_hash
                and existing["chunks_hash"] == chunks_hash
                and (existing["graph_indexed"] or not graph_indexed)
            ):
                return existing["id"]

        if extracted_chunks is None and graph_indexed:
            extracted_chunks = self.extract_document(chunks)
        if extracted_chunks is None:
            extracted_chunks = [
                {"entities": [], "relations": []} for _ in chunks
            ]
        if len(extracted_chunks) != len(chunks):
            raise ValueError("Le nombre de résultats d'extraction ne correspond pas aux passages.")

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE category = ? AND filename = ?",
                (category, filename)
            ).fetchone()
            document_id = existing["id"] if existing else uuid.uuid4().hex
            if existing:
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                connection.execute(
                    "UPDATE documents SET content_hash = ?, chunks_hash = ?, graph_indexed = ? "
                    "WHERE id = ?",
                    (content_hash, chunks_hash, int(graph_indexed), document_id)
                )
            else:
                connection.execute(
                    "INSERT INTO documents "
                    "(id, category, filename, content_hash, chunks_hash, graph_indexed) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (document_id, category, filename, content_hash, chunks_hash, int(graph_indexed))
                )

            self._insert_graph_records(connection, document_id, chunks, extracted_chunks)
            self._remove_orphan_entities(connection)
        return document_id

    def replace_document_graph(self, document_id, text, chunks, extracted_chunks):
        """Atomically replace graph records if the source document is unchanged."""
        if len(extracted_chunks) != len(chunks):
            raise ValueError("Le nombre de résultats d'extraction ne correspond pas aux passages.")
        content_hash = self._hash(text)
        chunks_hash = self._chunks_hash(chunks)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT content_hash, chunks_hash, graph_indexed "
                "FROM documents WHERE id = ?",
                (document_id,)
            ).fetchone()
            if (
                row is None
                or row["content_hash"] != content_hash
                or row["chunks_hash"] != chunks_hash
                or row["graph_indexed"]
            ):
                return False

            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._insert_graph_records(connection, document_id, chunks, extracted_chunks)
            connection.execute(
                "UPDATE documents SET graph_indexed = 1 WHERE id = ?",
                (document_id,)
            )
            self._remove_orphan_entities(connection)
        return True

    @staticmethod
    def _insert_graph_records(connection, document_id, chunks, extracted_chunks):
        for index, (chunk, graph_data) in enumerate(zip(chunks, extracted_chunks)):
            chunk_id = f"{document_id}:{index}"
            connection.execute(
                "INSERT INTO chunks (id, document_id, chunk_index, text) VALUES (?, ?, ?, ?)",
                (chunk_id, document_id, index, chunk)
            )
            entity_ids = {}
            for entity in graph_data["entities"]:
                normalized = " ".join(entity["name"].split()).casefold()
                connection.execute(
                    "INSERT OR IGNORE INTO entities (name, normalized_name, entity_type) "
                    "VALUES (?, ?, ?)",
                    (entity["name"], normalized, entity["type"])
                )
                row = connection.execute(
                    "SELECT id FROM entities WHERE normalized_name = ? AND entity_type = ?",
                    (normalized, entity["type"])
                ).fetchone()
                entity_ids[entity["name"].casefold()] = row["id"]
                connection.execute(
                    "INSERT OR IGNORE INTO mentions (chunk_id, entity_id) VALUES (?, ?)",
                    (chunk_id, row["id"])
                )

            for relation in graph_data["relations"]:
                source_id = entity_ids.get(relation["source"].casefold())
                target_id = entity_ids.get(relation["target"].casefold())
                if source_id is None or target_id is None or source_id == target_id:
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO relations "
                    "(source_entity_id, target_entity_id, relation_type, chunk_id) "
                    "VALUES (?, ?, ?, ?)",
                    (source_id, target_id, relation["type"], chunk_id)
                )

    @staticmethod
    def _remove_orphan_entities(connection):
        connection.execute(
            "DELETE FROM entities WHERE NOT EXISTS "
            "(SELECT 1 FROM mentions WHERE mentions.entity_id = entities.id)"
        )

    def get_document_id(self, category, filename):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM documents WHERE category = ? AND filename = ?",
                (category, filename)
            ).fetchone()
        return row["id"] if row else None

    def remove_document(self, category, filename):
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE category = ? AND filename = ?",
                (category, filename)
            )
            self._remove_orphan_entities(connection)

    def move_document(self, old_category, filename, new_category):
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET category = ? WHERE category = ? AND filename = ?",
                (new_category, old_category, filename)
            )

    def rename_category(self, old_name, new_name):
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET category = ? WHERE category = ?",
                (new_name, old_name)
            )

    def retain_documents(self, documents):
        """Drop graph rows whose source files are no longer present on disk."""
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, category, filename FROM documents"
            ).fetchall()
            valid = set(documents)
            stale_ids = [
                row["id"] for row in existing
                if (row["category"], row["filename"]) not in valid
            ]
            connection.executemany(
                "DELETE FROM documents WHERE id = ?",
                [(document_id,) for document_id in stale_ids]
            )
            self._remove_orphan_entities(connection)

    def expand_from_chunks(self, seed_chunk_ids, limit):
        """Find chunks sharing seed entities or entities linked to them."""
        seed_chunk_ids = list(dict.fromkeys(seed_chunk_ids))
        if not seed_chunk_ids or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in seed_chunk_ids)
        with self._connect() as connection:
            seed_entities = connection.execute(
                f"SELECT DISTINCT e.id, e.name FROM entities e "
                f"JOIN mentions m ON m.entity_id = e.id "
                f"WHERE m.chunk_id IN ({placeholders})",
                seed_chunk_ids
            ).fetchall()
            if not seed_entities:
                return []

            seed_ids = {row["id"] for row in seed_entities}
            seed_names = {row["id"]: row["name"] for row in seed_entities}
            relations = connection.execute(
                "SELECT source_entity_id, target_entity_id, relation_type "
                "FROM relations WHERE source_entity_id IN ({0}) OR target_entity_id IN ({0})"
                .format(",".join("?" for _ in seed_ids)),
                [*seed_ids, *seed_ids]
            ).fetchall()
            expanded_ids = set(seed_ids)
            relation_by_entity = {}
            for relation in relations:
                source_id = relation["source_entity_id"]
                target_id = relation["target_entity_id"]
                if source_id in seed_ids:
                    expanded_ids.add(target_id)
                    relation_by_entity[target_id] = (
                        seed_names[source_id], relation["relation_type"]
                    )
                if target_id in seed_ids:
                    expanded_ids.add(source_id)
                    relation_by_entity[source_id] = (
                        seed_names[target_id], relation["relation_type"]
                    )

            entity_placeholders = ",".join("?" for _ in expanded_ids)
            rows = connection.execute(
                f"SELECT c.id AS chunk_id, c.text AS chunk_text, "
                f"d.filename, d.category, e.id AS entity_id, e.name AS entity_name "
                f"FROM chunks c JOIN documents d ON d.id = c.document_id "
                f"JOIN mentions m ON m.chunk_id = c.id "
                f"JOIN entities e ON e.id = m.entity_id "
                f"WHERE e.id IN ({entity_placeholders}) "
                f"AND c.id NOT IN ({placeholders}) "
                f"ORDER BY d.category, d.filename, c.chunk_index",
                [*expanded_ids, *seed_chunk_ids]
            ).fetchall()

        candidates = {}
        for row in rows:
            candidate = candidates.setdefault(row["chunk_id"], {
                "chunk_id": row["chunk_id"],
                "filename": row["filename"],
                "category": row["category"],
                "chunk_text": row["chunk_text"],
                "matched_entities": [],
                "graph_links": []
            })
            if row["entity_name"] not in candidate["matched_entities"]:
                candidate["matched_entities"].append(row["entity_name"])
                entity_id = row["entity_id"]
                if entity_id in seed_ids:
                    candidate["graph_links"].append(
                        f"entité commune : {row['entity_name']}"
                    )
                elif entity_id in relation_by_entity:
                    seed_name, relation_type = relation_by_entity[entity_id]
                    candidate["graph_links"].append(
                        f"{seed_name} {relation_type} {row['entity_name']}"
                    )

        return sorted(
            candidates.values(),
            key=lambda candidate: (
                -len(candidate["matched_entities"]),
                candidate["category"],
                candidate["filename"]
            )
        )[:limit]
