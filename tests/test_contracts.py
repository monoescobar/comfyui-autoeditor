import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def class_assignments(filename, class_name):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    names = set()
    for item in node.body:
        if isinstance(item, ast.Assign):
            names.update(target.id for target in item.targets if isinstance(target, ast.Name))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
    return names


class StaticContractTests(unittest.TestCase):
    def test_all_python_files_compile(self):
        for path in ROOT.glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_user_facing_classes_have_help_contracts(self):
        classes = {
            "auto_editor.py": "DJ_AutoEditor",
            "audio_mixer.py": "DJ_AudioMixer",
            "lyrics_overlay.py": "DJ_LyricsOverlay",
        }
        required = {"DESCRIPTION", "OUTPUT_TOOLTIPS", "RETURN_TYPES", "RETURN_NAMES", "CATEGORY", "FUNCTION"}
        for filename, class_name in classes.items():
            self.assertTrue(required.issubset(class_assignments(filename, class_name)), class_name)

    def test_registered_type_names_and_compatibility_alias(self):
        combined = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in (
            "auto_editor.py", "audio_mixer.py", "lyrics_overlay.py"
        ))
        found = set(re.findall(r'^\s*"(DJ_[A-Za-z]+)":\s*DJ_', combined, flags=re.MULTILINE))
        self.assertEqual(found, {"DJ_AutoEditor", "DJ_AutoDirector", "DJ_AudioMixer", "DJ_LyricsOverlay"})

    def test_version_is_consistent(self):
        init_text = (ROOT / "__init__.py").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('__version__ = "2026.8.25.1"', init_text)
        self.assertIn('version = "2026.8.25.1"', pyproject)
        self.assertIn('AUTOEDITOR_NODE_VERSION = "v2026.08.25.1"', (ROOT / "auto_editor.py").read_text(encoding="utf-8"))
        self.assertIn('LYRICS_OVERLAY_NODE_VERSION = "v2026.08.25.1"', (ROOT / "lyrics_overlay.py").read_text(encoding="utf-8"))

    def test_documented_files_exist(self):
        for relative in (
            "README.md", "LICENSE", "CHANGELOG.md", "docs/NODE_REFERENCE.md",
            "docs/ARCHITECTURE.md", "docs/TROUBLESHOOTING.md", "requirements-optional.txt",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_no_workstation_paths_are_committed(self):
        for path in list(ROOT.glob("*.py")) + list((ROOT / "docs").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("E:\\\\ComfyUI", text, str(path))
            self.assertNotIn("C:\\\\Users\\\\mono", text, str(path))


if __name__ == "__main__":
    unittest.main()
