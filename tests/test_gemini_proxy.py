import unittest

from fastapi.testclient import TestClient

import main


OPENWHISPR_CLEANUP_PROMPT = (
    '* Input: "开箱。" (Simplified Chinese) * Role: Text cleanup tool for transcribed '
    "speech. * Task: Clean up transcribed text, fix grammar/punctuation, remove "
    "fillers, convert to Traditional Chinese (as per language context). * Constraint: "
    'Output ONLY cleaned text. No commentary. * "开箱" means "unboxing". * It Is a '
    'very short phrase. * Convert Simplified Chinese "开箱" to Traditional Chinese '
    '"開箱". * Keep the punctuation if it makes sense, or just the text. * 开箱。'
)

OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY = (
    '* Input: "我是认真的。测试一下啦，你看它上面语音辨识出问题。" '
    "* Task: Text cleanup of transcribed speech. * Constraints: * Remove filler words. "
    "* Fix grammar, spelling, punctuation. * Remove false starts/stutters. "
    "* Preserve voice/tone/intent. * Output ONLY cleaned text. "
    '* Language: Traditional Chinese. * "我是认真的。" (I\'m serious.) - Clear. '
    '* "测试一下啦，" (Just testing it.) - Clear. '
    '* "你看它上面语音辨识出问题。" (Look, the speech recognition on it has a problem.) - Clear. '
    "* The input is already quite clean, but it's in Simplified Chinese. "
    "* The system prompt requires Traditional Chinese output. "
    "* Simplified: 我是认真的。测试一下啦，你看它上面语音辨识出问题。 "
    "* Traditional: 我是認真的。測試一下啦，你看它上面語音辨識出問題。 "
    "* No filler words to remove. * No false starts. * Punctuation is okay. "
    "* Convert to Traditional Chinese.我是認真的。測試一下啦，你看它上面語音辨識出問題。"
)

OPENWHISPR_CLEANUP_PROMPT_GOAL_ARROW = """*   Input: "现在测试，这样正常。" (Now testing, this is normal.)
    *   Role: Text cleanup tool for transcribed speech.
    *   Goal: Clean up filler words, fix grammar/punctuation, remove false starts, preserve voice/intent, output ONLY cleaned text in Traditional Chinese.

    *   "现在测试，这样正常。"
    *   It's a simple statement. No filler words, no stutters, no false starts.
    *   Language: Simplified Chinese.
    *   Target Language: Traditional Chinese.

    *   "现在测试，这样正常。" $\\rightarrow$ "現在測試，這樣正常。"

    *   Output ONLY the cleaned text.
    *   No commentary.現在測試，這樣正常。"""


class GeminiProxyConversionTests(unittest.TestCase):
    def test_fallback_google_text_model_list_includes_current_gemini_and_gemma_models(self):
        ids = {model["id"] for model in main._google_text_model_entries()}

        self.assertIn("gemini-3.5-flash", ids)
        self.assertIn("gemini-3.1-pro-preview", ids)
        self.assertIn("gemini-3.1-flash-lite", ids)
        self.assertIn("gemini-2.5-flash", ids)
        self.assertIn("gemini-2.5-pro", ids)
        self.assertIn("gemma-4-31b-it", ids)
        self.assertIn("gemma-4-26b-a4b-it", ids)

    def test_filters_google_models_list_to_generate_content_models(self):
        models = main._google_text_models_from_listing(
            {
                "models": [
                    {
                        "name": "models/gemini-3.5-flash",
                        "displayName": "Gemini 3.5 Flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-embedding-001",
                        "displayName": "Gemini Embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/gemini-3.1-flash-tts-preview",
                        "displayName": "Gemini TTS",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            }
        )

        self.assertEqual(
            models,
            [
                {
                    "id": "gemini-3.5-flash",
                    "object": "model",
                    "owned_by": "google",
                    "display_name": "Gemini 3.5 Flash",
                }
            ],
        )

    def test_normalizes_models_prefix_from_openwhispr(self):
        self.assertEqual(
            main._normalize_gemini_model("models/gemma-4-31b-it"),
            "gemma-4-31b-it",
        )

    def test_converts_openai_messages_to_gemini_payload(self):
        payload = main._openai_chat_to_gemini_payload(
            {
                "messages": [
                    {"role": "system", "content": "Clean up dictation only."},
                    {"role": "user", "content": "呃 幫我 修正 標點"},
                ],
                "temperature": 0.2,
                "max_tokens": 256,
            }
        )

        self.assertEqual(
            payload["system_instruction"],
            {"parts": [{"text": "Clean up dictation only."}]},
        )
        self.assertEqual(
            payload["contents"],
            [{"role": "user", "parts": [{"text": "呃 幫我 修正 標點"}]}],
        )
        self.assertEqual(payload["generationConfig"]["temperature"], 0.2)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 256)

    def test_converts_openwhispr_cleanup_prompt_to_plain_transcript_input(self):
        payload = main._openai_chat_to_gemini_payload(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": OPENWHISPR_CLEANUP_PROMPT,
                    }
                ]
            }
        )

        self.assertEqual(
            payload["contents"],
            [{"role": "user", "parts": [{"text": "开箱。"}]}],
        )
        system_text = payload["system_instruction"]["parts"][0]["text"]
        self.assertIn("Output ONLY the cleaned text", system_text)
        self.assertIn("Convert Simplified Chinese to Traditional Chinese", system_text)
        self.assertNotIn("Input:", system_text)

    def test_converts_task_only_openwhispr_cleanup_prompt_to_plain_transcript_input(self):
        state = {}
        payload = main._openai_chat_to_gemini_payload(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY,
                    }
                ]
            },
            cleanup_state=state,
        )

        self.assertEqual(
            payload["contents"],
            [
                {
                    "role": "user",
                    "parts": [{"text": "我是认真的。测试一下啦，你看它上面语音辨识出问题。"}],
                }
            ],
        )
        self.assertEqual(state["input"], "我是认真的。测试一下啦，你看它上面语音辨识出问题。")

    def test_converts_goal_arrow_openwhispr_cleanup_prompt_to_plain_transcript_input(self):
        state = {}
        payload = main._openai_chat_to_gemini_payload(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": OPENWHISPR_CLEANUP_PROMPT_GOAL_ARROW,
                    }
                ]
            },
            cleanup_state=state,
        )

        self.assertEqual(
            payload["contents"],
            [{"role": "user", "parts": [{"text": "现在测试，这样正常。"}]}],
        )
        self.assertEqual(state["input"], "现在测试，这样正常。")

    def test_sanitizes_leaked_openwhispr_cleanup_prompt_from_model_response(self):
        self.assertEqual(
            main._sanitize_cleanup_response(
                OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY,
                cleanup_input="我是认真的。测试一下啦，你看它上面语音辨识出问题。",
            ),
            "我是認真的。測試一下啦，你看它上面語音辨識出問題。",
        )

    def test_sanitizes_goal_arrow_openwhispr_cleanup_prompt_from_model_response(self):
        self.assertEqual(
            main._sanitize_cleanup_response(
                OPENWHISPR_CLEANUP_PROMPT_GOAL_ARROW,
                cleanup_input="现在测试，这样正常。",
            ),
            "現在測試，這樣正常。",
        )

    def test_converts_gemini_response_to_openai_chat_response(self):
        response = main._gemini_response_to_openai_chat(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "幫我修正標點。"},
                                {"text": "\n"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 20,
                },
            },
            model="gemma-4-31b-it",
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["model"], "gemma-4-31b-it")
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(response["choices"][0]["message"]["content"], "幫我修正標點。\n")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["total_tokens"], 20)


class GeminiProxyEndpointTests(unittest.TestCase):
    def test_models_endpoint_includes_google_text_models(self):
        client = TestClient(main.app)
        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        ids = {model["id"] for model in response.json()["data"]}
        self.assertIn("Qwen/Qwen3-ASR-0.6B", ids)
        self.assertIn("gemini-3.5-flash", ids)
        self.assertIn("gemma-4-31b-it", ids)

    def test_chat_completions_uses_local_proxy_and_returns_openai_shape(self):
        async def fake_call(model, payload, api_key):
            self.assertEqual(model, "gemma-4-31b-it")
            self.assertEqual(api_key, "test-google-key")
            self.assertEqual(payload["contents"][0]["parts"][0]["text"], "hello")
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello cleaned"}]},
                        "finishReason": "STOP",
                    }
                ]
            }

        old_call = main._call_gemini_generate_content
        main._call_gemini_generate_content = fake_call
        try:
            client = TestClient(main.app)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-google-key"},
                json={
                    "model": "models/gemma-4-31b-it",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            main._call_gemini_generate_content = old_call

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model"], "gemma-4-31b-it")
        self.assertEqual(data["choices"][0]["message"]["content"], "hello cleaned")

    def test_chat_completions_does_not_forward_openwhispr_cleanup_prompt_as_user_text(self):
        async def fake_call(model, payload, api_key):
            self.assertEqual(payload["contents"][0]["parts"][0]["text"], "开箱。")
            self.assertNotIn("Input:", payload["contents"][0]["parts"][0]["text"])
            self.assertIn("Output ONLY the cleaned text", payload["system_instruction"]["parts"][0]["text"])
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "開箱。"}]},
                        "finishReason": "STOP",
                    }
                ]
            }

        old_call = main._call_gemini_generate_content
        main._call_gemini_generate_content = fake_call
        try:
            client = TestClient(main.app)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-google-key"},
                json={
                    "model": "gemma-4-31b-it",
                    "messages": [{"role": "user", "content": OPENWHISPR_CLEANUP_PROMPT}],
                },
            )
        finally:
            main._call_gemini_generate_content = old_call

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "開箱。")

    def test_chat_completions_strips_leaked_openwhispr_cleanup_prompt_response(self):
        async def fake_call(model, payload, api_key):
            self.assertEqual(
                payload["contents"][0]["parts"][0]["text"],
                "我是认真的。测试一下啦，你看它上面语音辨识出问题。",
            )
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY}]},
                        "finishReason": "STOP",
                    }
                ]
            }

        old_call = main._call_gemini_generate_content
        main._call_gemini_generate_content = fake_call
        try:
            client = TestClient(main.app)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-google-key"},
                json={
                    "model": "gemma-4-31b-it",
                    "messages": [{"role": "user", "content": OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY}],
                },
            )
        finally:
            main._call_gemini_generate_content = old_call

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "我是認真的。測試一下啦，你看它上面語音辨識出問題。",
        )

    def test_chat_completions_strips_goal_arrow_openwhispr_cleanup_prompt_response(self):
        async def fake_call(model, payload, api_key):
            self.assertEqual(payload["contents"][0]["parts"][0]["text"], "现在测试，这样正常。")
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": OPENWHISPR_CLEANUP_PROMPT_GOAL_ARROW}]},
                        "finishReason": "STOP",
                    }
                ]
            }

        old_call = main._call_gemini_generate_content
        main._call_gemini_generate_content = fake_call
        try:
            client = TestClient(main.app)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-google-key"},
                json={
                    "model": "gemma-4-31b-it",
                    "messages": [{"role": "user", "content": OPENWHISPR_CLEANUP_PROMPT_GOAL_ARROW}],
                },
            )
        finally:
            main._call_gemini_generate_content = old_call

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "現在測試，這樣正常。")

    def test_responses_endpoint_returns_openai_responses_shape(self):
        async def fake_call(model, payload, api_key):
            self.assertEqual(model, "gemma-4-31b-it")
            self.assertEqual(api_key, "test-google-key")
            self.assertEqual(payload["contents"][0]["parts"][0]["text"], "hello")
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello cleaned"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
            }

        old_call = main._call_gemini_generate_content
        main._call_gemini_generate_content = fake_call
        try:
            client = TestClient(main.app)
            response = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-google-key"},
                json={
                    "model": "gemma-4-31b-it",
                    "input": "hello",
                    "instructions": "Clean up dictation only.",
                },
            )
        finally:
            main._call_gemini_generate_content = old_call

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "response")
        self.assertEqual(data["output"][0]["content"][0]["text"], "hello cleaned")
        self.assertEqual(data["usage"]["total_tokens"], 5)

    def test_responses_endpoint_strips_leaked_openwhispr_cleanup_prompt_response(self):
        async def fake_call(model, payload, api_key):
            self.assertEqual(
                payload["contents"][0]["parts"][0]["text"],
                "我是认真的。测试一下啦，你看它上面语音辨识出问题。",
            )
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
            }

        old_call = main._call_gemini_generate_content
        main._call_gemini_generate_content = fake_call
        try:
            client = TestClient(main.app)
            response = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-google-key"},
                json={
                    "model": "gemma-4-31b-it",
                    "input": OPENWHISPR_CLEANUP_PROMPT_TASK_ONLY,
                },
            )
        finally:
            main._call_gemini_generate_content = old_call

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["output"][0]["content"][0]["text"],
            "我是認真的。測試一下啦，你看它上面語音辨識出問題。",
        )


if __name__ == "__main__":
    unittest.main()
