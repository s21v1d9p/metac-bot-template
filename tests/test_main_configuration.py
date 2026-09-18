import ast
import unittest
from pathlib import Path


class MainConfigurationTests(unittest.TestCase):
    def test_openrouter_models_use_zero_cost_configuration(self) -> None:
        source = Path(__file__).parents[1].joinpath("main.py").read_text()
        module = ast.parse(source)
        bot_call = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SummerTemplateBot2026"
        )
        llms_keyword = next(
            (keyword for keyword in bot_call.keywords if keyword.arg == "llms"),
            None,
        )

        self.assertIsNotNone(llms_keyword, "Bot must explicitly configure its LLMs")
        self.assertIsInstance(llms_keyword.value, ast.Dict)

        llms = {
            key.value: value
            for key, value in zip(
                llms_keyword.value.keys,
                llms_keyword.value.values,
                strict=True,
            )
            if isinstance(key, ast.Constant)
        }
        expected_models = {
            "default": "openrouter/qwen/qwen3.8-27b:free",
            "summarizer": "openrouter/qwen/qwen3.8-27b:free",
            "parser": "openrouter/qwen/qwen3.8-27b:free",
        }

        for purpose, expected_model in expected_models.items():
            with self.subTest(purpose=purpose):
                configured_model = llms.get(purpose)
                self.assertIsInstance(configured_model, ast.Call)

                model_keyword = next(
                    (
                        keyword
                        for keyword in configured_model.keywords
                        if keyword.arg == "model"
                    ),
                    None,
                )
                self.assertIsNotNone(model_keyword)
                self.assertEqual(model_keyword.value.value, expected_model)

        self.assertEqual(llms["researcher"].value, "no_research")

        predictions_keyword = next(
            keyword
            for keyword in bot_call.keywords
            if keyword.arg == "predictions_per_research_report"
        )
        self.assertEqual(predictions_keyword.value.value, 1)

        summarize_keyword = next(
            (
                keyword
                for keyword in bot_call.keywords
                if keyword.arg == "enable_summarize_research"
            ),
            None,
        )
        self.assertIsNotNone(summarize_keyword)
        self.assertFalse(summarize_keyword.value.value)


if __name__ == "__main__":
    unittest.main()
