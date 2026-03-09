"""Tests for repository structure."""

from pathlib import Path


class TestRepoStructure:
    """Test that required directories exist."""

    def test_src_directory_exists(self) -> None:
        """Verify src/ directory exists."""
        repo_root = Path(__file__).parent.parent
        src_dir = repo_root / "src"
        assert src_dir.exists(), "src/ directory must exist"
        assert src_dir.is_dir(), "src/ must be a directory"

    def test_tests_directory_exists(self) -> None:
        """Verify tests/ directory exists."""
        repo_root = Path(__file__).parent.parent
        tests_dir = repo_root / "tests"
        assert tests_dir.exists(), "tests/ directory must exist"
        assert tests_dir.is_dir(), "tests/ must be a directory"

    def test_data_directory_exists(self) -> None:
        """Verify data/ directory exists."""
        repo_root = Path(__file__).parent.parent
        data_dir = repo_root / "data"
        assert data_dir.exists(), "data/ directory must exist"
        assert data_dir.is_dir(), "data/ must be a directory"

    def test_src_init_exists(self) -> None:
        """Verify src/__init__.py exists."""
        repo_root = Path(__file__).parent.parent
        init_file = repo_root / "src" / "__init__.py"
        assert init_file.exists(), "src/__init__.py must exist"

    def test_tests_init_exists(self) -> None:
        """Verify tests/__init__.py exists."""
        repo_root = Path(__file__).parent.parent
        init_file = repo_root / "tests" / "__init__.py"
        assert init_file.exists(), "tests/__init__.py must exist"

    def test_readme_exists(self) -> None:
        """Verify README.md exists."""
        repo_root = Path(__file__).parent.parent
        readme = repo_root / "README.md"
        assert readme.exists(), "README.md must exist"

    def test_gitignore_exists(self) -> None:
        """Verify .gitignore exists."""
        repo_root = Path(__file__).parent.parent
        gitignore = repo_root / ".gitignore"
        assert gitignore.exists(), ".gitignore must exist"

    def test_pyproject_toml_exists(self) -> None:
        """Verify pyproject.toml exists."""
        repo_root = Path(__file__).parent.parent
        pyproject = repo_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml must exist"

    def test_github_workflows_directory_exists(self) -> None:
        """Verify .github/workflows/ directory exists."""
        repo_root = Path(__file__).parent.parent
        workflows_dir = repo_root / ".github" / "workflows"
        assert workflows_dir.exists(), ".github/workflows/ directory must exist"
        assert workflows_dir.is_dir(), ".github/workflows/ must be a directory"

    def test_ci_yml_exists(self) -> None:
        """Verify CI workflow file exists."""
        repo_root = Path(__file__).parent.parent
        ci_yml = repo_root / ".github" / "workflows" / "ci.yml"
        assert ci_yml.exists(), ".github/workflows/ci.yml must exist"
