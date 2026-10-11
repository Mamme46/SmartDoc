import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from graph_extraction import extract_document_graph
from llm import infer_document_category


class DocumentCategoryTests(unittest.TestCase):
    def test_meeting_notes_are_not_classified_as_recipes(self):
        text = (
            "Note de réunion — Projet Atlas. L'équipe examine le retard "
            "du déploiement dans les agences régionales."
        )
        self.assertEqual(infer_document_category(text, "01_note_reunion.pdf"), "Comptes rendus")

    def test_supplier_email_and_expense_note_receive_document_type_labels(self):
        supplier_email = (
            "Échange de courriels — Livraison et test. De : achats. "
            "Le rapport de test confirme la compatibilité du matériel."
        )
        self.assertEqual(
            infer_document_category(
                supplier_email,
                "03_courriel_fournisseur.pdf"
            ),
            "Courriels fournisseurs"
        )
        self.assertEqual(
            infer_document_category("Note de frais — déplacement professionnel"),
            "Notes de frais"
        )

    def test_recipe_classification_requires_recipe_context(self):
        self.assertIsNone(
            infer_document_category(
                "Une recette efficace pour la réussite du projet consiste à planifier."
            )
        )
        self.assertEqual(
            infer_document_category(
                "Recette de soupe. Ingrédients : tomates et eau. "
                "Préparation : cuire au four pendant 20 minutes."
            ),
            "Recettes"
        )

    def test_graph_extraction_batches_all_chunks_into_one_model_call(self):
        response = {
            "message": {
                "content": (
                    '{"entities": ['
                    '{"name":"Camille Durand","type":"person","passages":[0]},'
                    '{"name":"NovaTech","type":"organization","passages":[1]}'
                    '],"relations": ['
                    '{"source":"Camille Durand","target":"NovaTech",'
                    '"type":"travaille pour","passage":0}]}'
                )
            }
        }
        with patch("graph_extraction.ollama.chat", return_value=response) as chat:
            result = extract_document_graph([
                "Camille Durand travaille chez NovaTech.",
                "NovaTech développe le Projet Atlas."
            ])

        chat.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["entities"][0]["name"], "Camille Durand")
        self.assertEqual(result[0]["relations"][0]["type"], "travaille pour")
        self.assertEqual(result[1]["entities"][0]["name"], "NovaTech")


if __name__ == "__main__":
    unittest.main()
