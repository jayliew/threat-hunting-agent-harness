"""Config-driven model validation, including models absent from the built-in registry."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

from model_config import installed_model_error, load_model_formats


class ModelConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formats.toml"
            path.write_text(text)
            return load_model_formats(path)

    def test_new_renderer_and_parser_require_only_configuration(self):
        formats = self.load('''
[[formats]]
label = "Future Model"
name_pattern = "future-model"
renderers = ["future-renderer"]
parsers = ["future-parser"]
required_stops = ["<end>"]
allowed_stops = ["<end>", "<turn>"]
''')
        info = SimpleNamespace(
            template="{{ .Prompt }}",
            modelfile="RENDERER future-renderer\nPARSER future-parser\n",
            parameters='stop "<end>"\nstop "<turn>"\n',
        )
        name = "hf.co/vendor/Future-Model:latest"
        self.assertIsNone(installed_model_error(name, info, formats=formats))
        info.modelfile = "RENDERER future-renderer\n"
        self.assertIn("expects PARSER future-parser", installed_model_error(name, info, formats=formats))
        info.modelfile += "PARSER future-parser\n"
        info.parameters = 'stop "<turn>"'
        self.assertIn("missing required stops", installed_model_error(name, info, formats=formats))
        info.parameters = 'stop "<end>"\nstop "arbitrary text"'
        self.assertIn("unexpected stops", installed_model_error(name, info, formats=formats))

    def test_required_stop_can_allow_additional_stops(self):
        formats = self.load('''
[[formats]]
label = "Other Model"
name_pattern = "other-model"
markers = ["<user>", "<assistant>"]
required_stops = ["<end>"]
''')
        info = SimpleNamespace(
            template="<user>{{ .Content }}<assistant>",
            modelfile="PARSER some-parser\n",
            parameters='stop "<end>"\nstop "<other>"\n',
        )
        self.assertIsNone(installed_model_error("other-model", info, formats=formats))

    def test_multiline_prompt_text_is_not_a_modelfile_instruction(self):
        info = SimpleNamespace(
            template="<|system|>\n<|user|>\n<|assistant|>\n",
            modelfile='TEMPLATE """<|system|>\nPARSER wrong\nRENDERER wrong\n"""\n',
            parameters='stop "<|end_of_text|>"\n',
        )
        self.assertIsNone(installed_model_error("foundation-sec-8b-instruct", info))
        info.modelfile += "PARSER\n"
        self.assertIn("malformed Modelfile", installed_model_error("foundation-sec-8b-instruct", info))

    def test_escaped_stop_sequences_match_their_actual_values(self):
        formats = self.load('''
[[formats]]
label = "Escaped Stops"
name_pattern = "escaped"
markers = ["<user>"]
required_stops = ["\\nuser:", "\\u2603"]
''')
        info = SimpleNamespace(
            template="<user>", modelfile="",
            parameters='stop ' + json.dumps('\nuser:') + '\nstop ' + json.dumps('☃'),
        )
        self.assertIsNone(installed_model_error("escaped", info, formats=formats))

    def test_invalid_rules_fail_instead_of_disabling_validation(self):
        invalid_rules = [
            "",
            '[[formats]]\nlabel="X"\nname_pattern="x"\nmarker=["x"]',
            '[[formats]]\nlabel="X"\nname_pattern="x"',
            '[[formats]]\nlabel="X"\nname_pattern="["\nmarkers=["x"]',
            '[[formats]]\nlabel="X"\nname_pattern="x"\nmarkers="x"',
            '[[formats]]\nlabel="X"\nname_pattern="x"\nmarkers=["x"]\nrenderers=["x"]',
            '[[formats]]\nlabel="X"\nname_pattern="x"\nmarkers=["x"]\nrequired_stops=["end"]\nallowed_stops=[]',
        ]
        for text in invalid_rules:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)


if __name__ == "__main__":
    unittest.main()
