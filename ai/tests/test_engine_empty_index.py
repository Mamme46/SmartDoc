import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from engine import SmartDocEngine


class EmptyEmbeddingModel:
    @staticmethod
    def get_embedding_dimension():
        return 3

    @staticmethod
    def encode(texts, normalize_embeddings=True):
        return np.ones((len(texts), 3))


class EmptyEngineIndexTests(unittest.TestCase):
    def test_empty_index_retains_columns_needed_by_first_import(self):
        with tempfile.TemporaryDirectory() as data_path:
            engine = SmartDocEngine(data_path, EmptyEmbeddingModel())

            self.assertIn("category", engine.df.columns)
            self.assertIn("filename", engine.df.columns)
            self.assertIn("category", engine.chunks_df.columns)
            self.assertIn("filename", engine.chunks_df.columns)

            engine._remove_existing_document("New folder", "first.txt")

            self.assertTrue(engine.df.empty)
            self.assertTrue(engine.chunks_df.empty)

    def test_existing_document_load_assigns_stable_text_ids(self):
        with tempfile.TemporaryDirectory() as workspace:
            category_path = Path(workspace) / "Existing"
            category_path.mkdir()
            (category_path / "existing.txt").write_text(
                "An existing document.", encoding="utf-8"
            )

            with patch(
                "engine.KnowledgeGraph.extract_document",
                side_effect=lambda chunks: [
                    {"entities": [], "relations": []} for _ in chunks
                ],
            ), patch.object(SmartDocEngine, "_schedule_graph_index"):
                engine = SmartDocEngine(workspace, EmptyEmbeddingModel())

            self.assertEqual(len(engine.df), 1)
            self.assertIsInstance(engine.df.iloc[0]["document_id"], str)
            self.assertIsInstance(engine.chunks_df.iloc[0]["document_id"], str)
            self.assertTrue(engine.chunks_df.iloc[0]["chunk_id"].startswith(
                engine.df.iloc[0]["document_id"]
            ))

    def test_first_import_succeeds_with_an_empty_index(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            data_path = workspace_path / "user"
            data_path.mkdir()
            source_path = workspace_path / "first.txt"
            source_path.write_text("A first test document.", encoding="utf-8")

            with patch(
                "engine.generate_category_name", return_value="Test category"
            ), patch.object(
                SmartDocEngine, "_schedule_graph_index"
            ) as schedule_graph:
                engine = SmartDocEngine(data_path, EmptyEmbeddingModel())
                result = engine.categorize(source_path)

            schedule_graph.assert_called_once()
            self.assertNotIn("error", result)
            self.assertEqual(len(engine.df), 1)
            self.assertEqual(engine.df.iloc[0]["filename"], "first.txt")
            self.assertEqual(len(engine.chunks_df), 1)

    def test_known_document_type_skips_slow_category_name_model_call(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            data_path = workspace_path / "user"
            data_path.mkdir()
            source_path = workspace_path / "meeting.txt"
            source_path.write_text(
                "Note de réunion. L'équipe examine le calendrier du projet.",
                encoding="utf-8"
            )

            with patch("engine.generate_category_name") as generate_name, patch.object(
                SmartDocEngine, "_schedule_graph_index"
            ):
                engine = SmartDocEngine(data_path, EmptyEmbeddingModel())
                result = engine.categorize(source_path)

            generate_name.assert_not_called()
            self.assertEqual(result["category"], "Comptes rendus")
            self.assertEqual(engine.df.iloc[0]["category"], "Comptes rendus")


if __name__ == "__main__":
    unittest.main()
