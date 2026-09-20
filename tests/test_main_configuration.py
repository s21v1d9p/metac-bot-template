import ast
import unittest
from pathlib import Path


class MainConfigurationTests(unittest.TestCase):
    def test_openrouter_models_use_zero_cost_configuration(self) -> None:
        source = Path(__file__).parents[1].joinpath("main.py").read_text()
        module = ast.parse(source)
        bot_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SummerTemplateBot2026"
        )
        validation_samples = next(
            node.value
            for node in bot_class.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_structure_output_validation_samples"
                for target in node.targets
            )
        )
        self.assertEqual(validation_samples.value, 1)

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
            "default": "openrouter/openrouter/free",
            "summarizer": "openrouter/openrouter/free",
            "parser": "openrouter/openrouter/free",
        }

        for purpose, expected_model in expected_models.items():
            with self.subTest(purpose=purpose):
                configured_model = llms.get(purpose)
                self.assertIsInstance(configured_model, ast.Call)

                kwargs = {
                    keyword.arg: keyword.value
                    for keyword in configured_model.keywords
                }
                self.assertEqual(kwargs["model"].value, expected_model)
                self.assertEqual(kwargs["timeout"].value, 120)
                self.assertEqual(kwargs["allowed_tries"].value, 3)

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

    def test_test_mode_limits_smoke_test_to_one_question(self) -> None:
        source = Path(__file__).parents[1].joinpath("main.py").read_text()
        module = ast.parse(source)

        question_slice = next(
            (
                node
                for node in ast.walk(module)
                if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "test_questions"
                and isinstance(node.slice, ast.Slice)
            ),
            None,
        )

        self.assertIsNotNone(question_slice)
        self.assertIsNone(question_slice.slice.lower)
        self.assertEqual(question_slice.slice.upper.value, 1)


if __name__ == "__main__":
    unittest.main()
