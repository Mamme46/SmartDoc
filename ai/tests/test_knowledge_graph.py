import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from knowledge_graph import KnowledgeGraph


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.graph = KnowledgeGraph(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_expands_across_documents_and_categories_through_relations(self):
        seed_id = self.graph.sync_document(
            "Contrats",
            "contrat.txt",
            "A travaille avec B.",
            ["A est cité ici."],
            extracted_chunks=[{
                "entities": [{"name": "A", "type": "organization"}],
                "relations": []
            }]
        )
        self.graph.sync_document(
            "Projets",
            "projet.txt",
            "A travaille avec B.",
            ["A travaille avec B."],
            extracted_chunks=[{
                "entities": [
                    {"name": "A", "type": "organization"},
                    {"name": "B", "type": "person"}
                ],
                "relations": [{"source": "A", "target": "B", "type": "collabore avec"}]
            }]
        )
        self.graph.sync_document(
            "Courriers",
            "courrier.txt",
            "B intervient dans le projet.",
            ["B intervient dans le projet."],
            extracted_chunks=[{
                "entities": [{"name": "B", "type": "person"}],
                "relations": []
            }]
        )

        related = self.graph.expand_from_chunks([f"{seed_id}:0"], limit=10)

        self.assertEqual(
            {(item["category"], item["filename"]) for item in related},
            {("Projets", "projet.txt"), ("Courriers", "courrier.txt")}
        )
        self.assertTrue(any("A collabore avec B" in item["graph_links"] for item in related))

    def test_preserves_document_id_when_content_is_updated(self):
        extracted = [{
            "entities": [{"name": "Alice", "type": "person"}],
            "relations": []
        }]
        first_id = self.graph.sync_document(
            "Dossier", "note.txt", "Alice.", ["Alice."], extracted
        )
        updated_id = self.graph.sync_document(
            "Dossier",
            "note.txt",
            "Alice travaille ici.",
            ["Alice travaille ici."],
            extracted
        )

        self.assertEqual(first_id, updated_id)
        self.assertFalse(
            self.graph.needs_reindex(
                "Dossier", "note.txt", "Alice travaille ici.", ["Alice travaille ici."]
            )
        )

    def test_background_graph_index_marks_pending_documents_complete(self):
        text = "Alice travaille chez NovaTech."
        chunks = [text]
        document_id = self.graph.sync_document(
            "Dossier",
            "note.txt",
            text,
            chunks,
            graph_indexed=False
        )
        self.assertTrue(self.graph.needs_reindex("Dossier", "note.txt", text, chunks))

        extracted = [{
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "NovaTech", "type": "organization"}
            ],
            "relations": [{
                "source": "Alice",
                "target": "NovaTech",
                "type": "travaille chez"
            }]
        }]
        self.assertTrue(
            self.graph.replace_document_graph(document_id, text, chunks, extracted)
        )
        self.assertFalse(self.graph.needs_reindex("Dossier", "note.txt", text, chunks))

    def test_stale_background_result_does_not_replace_new_document_content(self):
        first_text = "Alice travaille chez NovaTech."
        first_chunks = [first_text]
        document_id = self.graph.sync_document(
            "Dossier", "note.txt", first_text, first_chunks, graph_indexed=False
        )
        latest_text = "Alice travaille chez Orion."
        latest_chunks = [latest_text]
        self.graph.sync_document(
            "Dossier", "note.txt", latest_text, latest_chunks, graph_indexed=False
        )

        self.assertFalse(
            self.graph.replace_document_graph(
                document_id,
                first_text,
                first_chunks,
                [{"entities": [], "relations": []}]
            )
        )
        self.assertTrue(
            self.graph.needs_reindex("Dossier", "note.txt", latest_text, latest_chunks)
        )

    def test_graph_data_is_isolated_by_storage_path(self):
        self.graph.sync_document(
            "Dossier",
            "note.txt",
            "Alice.",
            ["Alice."],
            [{"entities": [{"name": "Alice", "type": "person"}], "relations": []}]
        )
        other_graph = KnowledgeGraph(Path(self.temp_dir.name) / "other-user")
        self.assertEqual(other_graph.expand_from_chunks(["unknown:0"], limit=10), [])
        self.assertIsNone(other_graph.get_document_id("Dossier", "note.txt"))


if __name__ == "__main__":
    unittest.main()
