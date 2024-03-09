# Libraries importing
import unittest
import json
from backend import app


# Unit test class
class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_answer_endpoint(self):
        response = self.app.post("/get_answer", data=json.dumps({"question": "What is Flask?"}),
                                 content_type="application/json")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", data)
        self.assertIn("faithfulness_score", data)
        self.assertIn("source_documents", data)

    def test_reload_base_endpoint(self):
        response = self.app.get("/reload_base")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["message"], "Base data reloaded successfully")

if __name__ == "__main__":
    unittest.main()
