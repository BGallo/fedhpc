import os
import pytest

# Run tests from the project root so relative paths to data/ work
@pytest.fixture(autouse=True)
def chdir_to_project_root(monkeypatch):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(root)
