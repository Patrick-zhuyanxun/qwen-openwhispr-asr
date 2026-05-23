import unittest

from fastapi.testclient import TestClient

import main


class ApiSurfaceTests(unittest.TestCase):
    def test_models_endpoint_lists_only_local_qwen_asr_model(self):
        client = TestClient(main.app)
        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "object": "list",
                "data": [
                    {
                        "id": "Qwen/Qwen3-ASR-0.6B",
                        "object": "model",
                        "owned_by": "qwen",
                        "display_name": "Qwen3-ASR",
                    }
                ],
            },
        )

    def test_language_model_proxy_endpoints_are_not_exposed(self):
        client = TestClient(main.app)

        for path in (
            "/chat/completions",
            "/v1/chat/completions",
            "/responses",
            "/v1/responses",
        ):
            with self.subTest(path=path):
                response = client.post(path, json={"model": "external-cleanup-model"})
                self.assertEqual(response.status_code, 404)

    def test_health_does_not_report_language_model_proxy_state(self):
        client = TestClient(main.app)
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("gemini_proxy", data)
        self.assertNotIn("default_gemini_model", data)


if __name__ == "__main__":
    unittest.main()
