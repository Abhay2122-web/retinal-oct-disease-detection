"""Check project structure, Python syntax and key dependencies before running."""
import ast
import importlib.util
from pathlib import Path

REQUIRED_FILES = [
    "config.py", "data_utils.py", "models.py", "gradcam.py", "train.py",
    "app.py", "requirements.txt",
]
OPTIONAL_FILES = ["oct_dataset.py", "compare_models.py", "save_sample_oct_images.py"]
DEPENDENCIES = ["torch", "torchvision", "datasets", "streamlit", "PIL", "numpy",
                "pandas", "matplotlib", "sklearn", "tqdm"]


def main():
    root = Path.cwd()
    print("Project folder:", root)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        print("Missing required files:", ", ".join(missing))
        print("Note: app.py is expected. If your file is named app(3).py, make a copy named app.py.")
    else:
        print("Required project files: OK")

    py_files = sorted(root.glob("*.py"))
    syntax_errors = []
    for path in py_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append((path.name, str(exc)))
    if syntax_errors:
        print("Python syntax issues:")
        for name, error in syntax_errors:
            print(f" - {name}: {error}")
    else:
        print(f"Python syntax: OK ({len(py_files)} top-level .py files checked)")

    for package in DEPENDENCIES:
        available = importlib.util.find_spec(package) is not None
        print(f"{'OK' if available else 'MISSING'} dependency: {package}")

    for folder in ("models", "outputs", "sample_oct_images"):
        (root / folder).mkdir(exist_ok=True)
    print("Output folders: ready")
    print("Verification checks file presence and syntax; it does not prove training accuracy or clinical validity.")


if __name__ == "__main__":
    main()
