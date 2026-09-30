"""Integration checks against the local trained weights, without network access."""

import io
import os
import unittest
from pathlib import Path

os.environ["USE_LOCAL_MODELS"] = "1"

from app import MODEL_INFOS, app

ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_page_contains_available_models(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        for model in MODEL_INFOS:
            self.assertIn(model.key.encode(), response.data)

    def test_all_models_can_classify_example_images(self):
        self.assertTrue(MODEL_INFOS)
        images = sorted((ROOT / "static/assets/examples").glob("*/*.jpg"))
        self.assertEqual(len(images), 6)
        for model in MODEL_INFOS:
            for image in images:
                with self.subTest(model=model.key, image=image.name):
                    response = self.client.post(
                        "/api/predict",
                        data={"model_key": model.key,
                              "image": (io.BytesIO(image.read_bytes()), image.name)},
                    )
                    self.assertEqual(response.status_code, 200)
                    result = response.get_json()
                    self.assertEqual(result["model"]["key"], model.key)
                    self.assertEqual(
                        {row["label"] for row in result["probabilities"]},
                        set(model.classes),
                    )
                    self.assertAlmostEqual(
                        sum(row["probability"] for row in result["probabilities"]), 1,
                        places=6,
                    )
                    for row in result["probabilities"]:
                        self.assertGreaterEqual(row["probability"], 0)
                        self.assertLessEqual(row["probability"], 1)
                    self.assertEqual(
                        result["prediction"],
                        max(result["probabilities"], key=lambda row: row["probability"]),
                    )

    def test_invalid_image_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            data={"model_key": MODEL_INFOS[0].key,
                  "image": (io.BytesIO(b"invalid"), "invalid.jpg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_unknown_model_is_rejected(self):
        response = self.client.post("/api/predict", data={"model_key": "unknown"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
