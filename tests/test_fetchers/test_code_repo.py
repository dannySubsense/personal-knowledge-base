"""Tests for the code repository fetcher."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from personal_knowledge_base.fetchers.base import FetchResult
from personal_knowledge_base.fetchers.code_repo import CodeRepoFetcher, RepoMetadata


class TestCodeRepoFetcher:
    """Tests for CodeRepoFetcher class."""

    @pytest.fixture
    def fetcher(self) -> CodeRepoFetcher:
        """Create a CodeRepoFetcher instance for testing."""
        return CodeRepoFetcher()

    @pytest.fixture
    def fetcher_with_token(self) -> CodeRepoFetcher:
        """Create a CodeRepoFetcher instance with GitHub token."""
        return CodeRepoFetcher(github_token="test_token_123")

    # can_fetch tests

    def test_can_fetch_github_https_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with standard GitHub HTTPS URL."""
        url = "https://github.com/user/repo"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_github_https_url_with_www(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with GitHub URL including www."""
        url = "https://www.github.com/user/repo"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_github_https_url_with_git_suffix(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with GitHub URL ending in .git."""
        url = "https://github.com/user/repo.git"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_github_ssh_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with GitHub SSH URL."""
        url = "git@github.com:user/repo.git"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_gitlab_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with GitLab URL."""
        url = "https://gitlab.com/user/repo"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_bitbucket_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with Bitbucket URL."""
        url = "https://bitbucket.org/user/repo"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_generic_git_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch with generic .git URL."""
        url = "https://git.example.com/repo.git"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_non_git_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch returns False for non-git URLs."""
        url = "https://example.com/page.html"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_youtube_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test can_fetch returns False for YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is False

    # _parse_github_repo tests

    def test_parse_github_repo_standard_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test parsing standard GitHub URL."""
        url = "https://github.com/user/repo"
        result = fetcher._parse_github_repo(url)
        assert result == ("user", "repo")

    def test_parse_github_repo_with_git_suffix(self, fetcher: CodeRepoFetcher) -> None:
        """Test parsing GitHub URL with .git suffix."""
        url = "https://github.com/user/repo.git"
        result = fetcher._parse_github_repo(url)
        assert result == ("user", "repo")

    def test_parse_github_repo_with_trailing_slash(self, fetcher: CodeRepoFetcher) -> None:
        """Test parsing GitHub URL with trailing slash."""
        url = "https://github.com/user/repo/"
        result = fetcher._parse_github_repo(url)
        assert result == ("user", "repo")

    def test_parse_github_repo_ssh_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test parsing GitHub SSH URL."""
        url = "git@github.com:user/repo.git"
        result = fetcher._parse_github_repo(url)
        assert result == ("user", "repo")

    def test_parse_github_repo_invalid_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test parsing non-GitHub URL returns None."""
        url = "https://gitlab.com/user/repo"
        result = fetcher._parse_github_repo(url)
        assert result is None

    # _fetch_github_metadata tests

    def test_fetch_github_metadata_success(self, fetcher: CodeRepoFetcher) -> None:
        """Test successful GitHub API metadata fetch."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stargazers_count": 100,
            "language": "Python",
            "description": "A test repo",
            "topics": ["python", "testing"],
            "license": {"name": "MIT License"},
            "forks_count": 50,
            "open_issues_count": 10,
            "default_branch": "main",
        }

        with patch.object(fetcher._get_session(), "get", return_value=mock_response):
            result = fetcher._fetch_github_metadata("user", "repo")

        assert result is not None
        assert result.stars == 100
        assert result.language == "Python"
        assert result.description == "A test repo"
        assert result.topics == ["python", "testing"]
        assert result.license == "MIT License"
        assert result.fork_count == 50
        assert result.open_issues == 10
        assert result.default_branch == "main"

    def test_fetch_github_metadata_not_found(self, fetcher: CodeRepoFetcher) -> None:
        """Test GitHub API returns 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(fetcher._get_session(), "get", return_value=mock_response):
            result = fetcher._fetch_github_metadata("user", "nonexistent")

        assert result is None

    def test_fetch_github_metadata_rate_limit(self, fetcher: CodeRepoFetcher) -> None:
        """Test GitHub API returns 403 (rate limit)."""
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch.object(fetcher._get_session(), "get", return_value=mock_response):
            result = fetcher._fetch_github_metadata("user", "repo")

        assert result is None

    def test_fetch_github_metadata_request_exception(self, fetcher: CodeRepoFetcher) -> None:
        """Test handling of request exception."""
        with patch.object(
            fetcher._get_session(), "get", side_effect=requests.RequestException("Network error")
        ):
            result = fetcher._fetch_github_metadata("user", "repo")

        assert result is None

    def test_fetch_github_metadata_with_auth_token(
        self, fetcher_with_token: CodeRepoFetcher
    ) -> None:
        """Test that auth token is included in headers."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 10}

        with patch.object(fetcher_with_token._get_session(), "get", return_value=mock_response):
            fetcher_with_token._fetch_github_metadata("user", "repo")

            # Verify the session has the auth header
            assert (
                fetcher_with_token._get_session().headers.get("Authorization")
                == "token test_token_123"
            )

    # _clone_repo tests

    def test_clone_repo_success(self, fetcher: CodeRepoFetcher) -> None:
        """Test successful git clone."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = fetcher._clone_repo("https://github.com/user/repo", "/tmp/test_repo")

        assert result is True
        mock_run.assert_called_once_with(
            ["git", "clone", "--depth", "1", "https://github.com/user/repo", "/tmp/test_repo"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_clone_repo_failure(self, fetcher: CodeRepoFetcher) -> None:
        """Test failed git clone."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128)
            result = fetcher._clone_repo("https://github.com/user/repo", "/tmp/test_repo")

        assert result is False

    def test_clone_repo_timeout(self, fetcher: CodeRepoFetcher) -> None:
        """Test git clone timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 120)):
            result = fetcher._clone_repo("https://github.com/user/repo", "/tmp/test_repo")

        assert result is False

    def test_clone_repo_git_not_found(self, fetcher: CodeRepoFetcher) -> None:
        """Test git command not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = fetcher._clone_repo("https://github.com/user/repo", "/tmp/test_repo")

        assert result is False

    # _find_readme tests

    def test_find_readme_md(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test finding README.md."""
        readme = tmp_path / "README.md"
        readme.write_text("# Test")
        result = fetcher._find_readme(str(tmp_path))
        assert result == readme

    def test_find_readme_rst(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test finding README.rst."""
        readme = tmp_path / "README.rst"
        readme.write_text("Test")
        result = fetcher._find_readme(str(tmp_path))
        assert result == readme

    def test_find_readme_txt(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test finding README.txt."""
        readme = tmp_path / "README.txt"
        readme.write_text("Test")
        result = fetcher._find_readme(str(tmp_path))
        assert result == readme

    def test_find_readme_no_extension(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test finding README without extension."""
        readme = tmp_path / "README"
        readme.write_text("Test")
        result = fetcher._find_readme(str(tmp_path))
        assert result == readme

    def test_find_readme_case_insensitive(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test finding readme.md (lowercase)."""
        readme = tmp_path / "readme.md"
        readme.write_text("# Test")
        result = fetcher._find_readme(str(tmp_path))
        assert result == readme

    def test_find_readme_not_found(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test when no README exists."""
        result = fetcher._find_readme(str(tmp_path))
        assert result is None

    def test_find_readme_preference_order(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test that README.md is preferred over README.rst."""
        readme_md = tmp_path / "README.md"
        readme_rst = tmp_path / "README.rst"
        readme_md.write_text("# Markdown")
        readme_rst.write_text("RST")
        result = fetcher._find_readme(str(tmp_path))
        assert result == readme_md

    # _read_readme tests

    def test_read_readme_success(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test reading README file."""
        readme = tmp_path / "README.md"
        content = "# Test Repository\n\nThis is a test."
        readme.write_text(content)
        result = fetcher._read_readme(readme)
        assert result == content

    def test_read_readme_unicode(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test reading README with unicode characters."""
        readme = tmp_path / "README.md"
        content = "# Test 🚀\n\nUnicode: 你好世界"
        readme.write_text(content, encoding="utf-8")
        result = fetcher._read_readme(readme)
        assert result == content

    def test_read_readme_binary_fallback(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test reading file with invalid UTF-8."""
        readme = tmp_path / "README.md"
        readme.write_bytes(b"# Test \xff\xfe\nContent")
        result = fetcher._read_readme(readme)
        assert "Test" in result

    def test_read_readme_io_error(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test handling of IO error when reading README."""
        readme = tmp_path / "README.md"
        readme.write_text("Test")
        readme.chmod(0o000)  # Remove read permissions
        try:
            result = fetcher._read_readme(readme)
            assert result == ""
        finally:
            readme.chmod(0o644)  # Restore permissions for cleanup

    # _get_repo_structure tests

    def test_get_repo_structure(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test getting repository structure."""
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "README.md").write_text("# Test")
        (tmp_path / "setup.py").write_text("")

        result = fetcher._get_repo_structure(str(tmp_path))
        assert "README.md" in result
        assert "setup.py" in result
        assert "src/" in result
        assert "tests/" in result

    def test_get_repo_structure_skips_hidden(
        self, fetcher: CodeRepoFetcher, tmp_path: Path
    ) -> None:
        """Test that hidden files are skipped."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".github").mkdir()
        (tmp_path / "README.md").write_text("# Test")

        result = fetcher._get_repo_structure(str(tmp_path))
        assert "README.md" in result
        assert ".git/" not in result
        assert ".github/" not in result

    def test_get_repo_structure_max_items(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test max_items limit."""
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")

        result = fetcher._get_repo_structure(str(tmp_path), max_items=5)
        assert len(result) == 5

    def test_get_repo_structure_empty_repo(self, fetcher: CodeRepoFetcher, tmp_path: Path) -> None:
        """Test empty repository."""
        result = fetcher._get_repo_structure(str(tmp_path))
        assert result == []

    # _extract_title_from_readme tests

    def test_extract_title_from_h1(self, fetcher: CodeRepoFetcher) -> None:
        """Test extracting title from H1 markdown."""
        content = "# My Project\n\nDescription here."
        result = fetcher._extract_title_from_readme(content)
        assert result == "My Project"

    def test_extract_title_from_h1_with_extra_spaces(self, fetcher: CodeRepoFetcher) -> None:
        """Test extracting title from H1 with extra spaces."""
        content = "#   My Project  \n\nDescription here."
        result = fetcher._extract_title_from_readme(content)
        assert result == "My Project"

    def test_extract_title_rst_style(self, fetcher: CodeRepoFetcher) -> None:
        """Test extracting title from RST-style heading."""
        content = "My Project\n==========\n\nDescription here."
        result = fetcher._extract_title_from_readme(content)
        assert result == "My Project"

    def test_extract_title_fallback_first_line(self, fetcher: CodeRepoFetcher) -> None:
        """Test fallback to first non-empty line."""
        content = "My Project Title\n\nDescription here."
        result = fetcher._extract_title_from_readme(content)
        assert result == "My Project Title"

    def test_extract_title_empty_readme(self, fetcher: CodeRepoFetcher) -> None:
        """Test empty README."""
        content = ""
        result = fetcher._extract_title_from_readme(content)
        assert result == ""

    def test_extract_title_whitespace_only(self, fetcher: CodeRepoFetcher) -> None:
        """Test README with only whitespace."""
        content = "   \n\n   "
        result = fetcher._extract_title_from_readme(content)
        assert result == ""

    def test_extract_title_truncated(self, fetcher: CodeRepoFetcher) -> None:
        """Test that long first lines are truncated."""
        content = "A" * 200
        result = fetcher._extract_title_from_readme(content)
        assert len(result) == 100

    # fetch tests - integration style

    def test_fetch_invalid_url(self, fetcher: CodeRepoFetcher) -> None:
        """Test fetch with invalid URL."""
        result = fetcher.fetch("https://example.com/page.html")
        assert result.success is False
        assert "Invalid repository URL" in result.error_message
        assert result.content_type == "code_repo"

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_fetch_github_metadata")
    def test_fetch_github_repo_not_found(
        self, mock_fetch_metadata: Mock, mock_parse_repo: Mock, fetcher: CodeRepoFetcher
    ) -> None:
        """Test fetch when GitHub repo returns 404."""
        mock_parse_repo.return_value = ("user", "nonexistent")
        mock_fetch_metadata.return_value = None

        result = fetcher.fetch("https://github.com/user/nonexistent")
        assert result.success is False
        assert "404" in result.error_message

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_fetch_github_metadata")
    @patch.object(CodeRepoFetcher, "_clone_repo")
    def test_fetch_clone_failure(
        self,
        mock_clone: Mock,
        mock_fetch_metadata: Mock,
        mock_parse_repo: Mock,
        fetcher: CodeRepoFetcher,
    ) -> None:
        """Test fetch when git clone fails."""
        mock_parse_repo.return_value = ("user", "repo")
        mock_fetch_metadata.return_value = RepoMetadata(stars=10)
        mock_clone.return_value = False

        result = fetcher.fetch("https://github.com/user/repo")
        assert result.success is False
        assert "Failed to clone" in result.error_message

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_fetch_github_metadata")
    @patch.object(CodeRepoFetcher, "_clone_repo")
    @patch.object(CodeRepoFetcher, "_find_readme")
    def test_fetch_no_readme(
        self,
        mock_find_readme: Mock,
        mock_clone: Mock,
        mock_fetch_metadata: Mock,
        mock_parse_repo: Mock,
        fetcher: CodeRepoFetcher,
    ) -> None:
        """Test fetch when no README exists."""
        mock_parse_repo.return_value = ("user", "repo")
        mock_fetch_metadata.return_value = RepoMetadata(stars=10)
        mock_clone.return_value = True
        mock_find_readme.return_value = None

        result = fetcher.fetch("https://github.com/user/repo")
        assert result.success is False
        assert "No README found" in result.error_message

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_fetch_github_metadata")
    @patch.object(CodeRepoFetcher, "_clone_repo")
    @patch.object(CodeRepoFetcher, "_find_readme")
    @patch.object(CodeRepoFetcher, "_read_readme")
    def test_fetch_empty_readme(
        self,
        mock_read_readme: Mock,
        mock_find_readme: Mock,
        mock_clone: Mock,
        mock_fetch_metadata: Mock,
        mock_parse_repo: Mock,
        fetcher: CodeRepoFetcher,
    ) -> None:
        """Test fetch when README is empty."""
        mock_parse_repo.return_value = ("user", "repo")
        mock_fetch_metadata.return_value = RepoMetadata(stars=10)
        mock_clone.return_value = True
        mock_find_readme.return_value = Path("/tmp/readme")
        mock_read_readme.return_value = "   "

        result = fetcher.fetch("https://github.com/user/repo")
        assert result.success is False
        assert "README is empty" in result.error_message

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_fetch_github_metadata")
    @patch.object(CodeRepoFetcher, "_clone_repo")
    @patch.object(CodeRepoFetcher, "_find_readme")
    @patch.object(CodeRepoFetcher, "_read_readme")
    @patch.object(CodeRepoFetcher, "_get_repo_structure")
    @patch("tempfile.mkdtemp")
    @patch("shutil.rmtree")
    def test_fetch_success(
        self,
        mock_rmtree: Mock,
        mock_mkdtemp: Mock,
        mock_get_structure: Mock,
        mock_read_readme: Mock,
        mock_find_readme: Mock,
        mock_clone: Mock,
        mock_fetch_metadata: Mock,
        mock_parse_repo: Mock,
        fetcher: CodeRepoFetcher,
    ) -> None:
        """Test successful fetch."""
        mock_parse_repo.return_value = ("user", "repo")
        mock_fetch_metadata.return_value = RepoMetadata(
            stars=100,
            language="Python",
            description="A test repo",
            topics=["python"],
            license="MIT",
            fork_count=50,
            open_issues=10,
            default_branch="main",
        )
        mock_clone.return_value = True
        mock_find_readme.return_value = Path("/tmp/repo/README.md")
        mock_read_readme.return_value = "# My Project\n\nThis is a test."
        mock_get_structure.return_value = ["README.md", "src/"]
        mock_mkdtemp.return_value = "/tmp/repo"

        result = fetcher.fetch("https://github.com/user/repo")

        assert result.success is True
        assert result.title == "My Project"
        assert result.content == "# My Project\n\nThis is a test."
        assert result.content_type == "code_repo"
        assert result.metadata["stars"] == 100
        assert result.metadata["language"] == "Python"
        assert result.metadata["description"] == "A test repo"
        assert result.metadata["topics"] == ["python"]
        assert result.metadata["license"] == "MIT"
        assert result.metadata["fork_count"] == 50
        assert result.metadata["open_issues"] == 10
        assert result.metadata["default_branch"] == "main"
        assert result.metadata["repo_structure"] == ["README.md", "src/"]
        assert result.metadata["github_owner"] == "user"
        assert result.metadata["github_repo"] == "repo"

        # Verify cleanup was called
        mock_rmtree.assert_called_once_with("/tmp/repo", ignore_errors=True)

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_clone_repo")
    @patch.object(CodeRepoFetcher, "_find_readme")
    @patch.object(CodeRepoFetcher, "_read_readme")
    @patch.object(CodeRepoFetcher, "_get_repo_structure")
    @patch("tempfile.mkdtemp")
    @patch("shutil.rmtree")
    def test_fetch_success_non_github(
        self,
        mock_rmtree: Mock,
        mock_mkdtemp: Mock,
        mock_get_structure: Mock,
        mock_read_readme: Mock,
        mock_find_readme: Mock,
        mock_clone: Mock,
        mock_parse_repo: Mock,
        fetcher: CodeRepoFetcher,
    ) -> None:
        """Test successful fetch from non-GitHub repo."""
        mock_parse_repo.return_value = None  # Not a GitHub repo
        mock_clone.return_value = True
        mock_find_readme.return_value = Path("/tmp/repo/README.md")
        mock_read_readme.return_value = "# GitLab Project\n\nThis is a test."
        mock_get_structure.return_value = ["README.md"]
        mock_mkdtemp.return_value = "/tmp/repo"

        result = fetcher.fetch("https://gitlab.com/user/repo")

        assert result.success is True
        assert result.title == "GitLab Project"
        assert result.metadata["repo_structure"] == ["README.md"]
        assert "stars" not in result.metadata

    @patch.object(CodeRepoFetcher, "_parse_github_repo")
    @patch.object(CodeRepoFetcher, "_fetch_github_metadata")
    @patch.object(CodeRepoFetcher, "_clone_repo")
    @patch("tempfile.mkdtemp")
    @patch("shutil.rmtree")
    def test_fetch_cleanup_on_clone_failure(
        self,
        mock_rmtree: Mock,
        mock_mkdtemp: Mock,
        mock_clone: Mock,
        mock_fetch_metadata: Mock,
        mock_parse_repo: Mock,
        fetcher: CodeRepoFetcher,
    ) -> None:
        """Test that temp directory is cleaned up even on clone failure."""
        mock_parse_repo.return_value = ("user", "repo")
        mock_fetch_metadata.return_value = RepoMetadata(stars=10)
        mock_clone.return_value = False
        mock_mkdtemp.return_value = "/tmp/repo"

        result = fetcher.fetch("https://github.com/user/repo")

        assert result.success is False
        mock_rmtree.assert_called_once_with("/tmp/repo", ignore_errors=True)

    # fetch_with_add_to_queue tests

    @patch("personal_knowledge_base.queue.operations.add_job")
    @patch.object(CodeRepoFetcher, "fetch")
    def test_fetch_with_add_to_queue_success(
        self, mock_fetch: Mock, mock_add_job: Mock, fetcher: CodeRepoFetcher
    ) -> None:
        """Test fetch_with_add_to_queue on successful fetch."""
        mock_fetch.return_value = FetchResult(
            url="https://github.com/user/repo",
            title="Test",
            content="Content",
            content_type="code_repo",
            success=True,
        )

        result = fetcher.fetch_with_add_to_queue(
            "https://github.com/user/repo", priority=1, kb_name="my_kb"
        )

        assert result.success is True
        mock_add_job.assert_called_once_with(
            url="https://github.com/user/repo",
            priority=1,
            content_type="code_repo",
            kb_name="my_kb",
        )

    @patch("personal_knowledge_base.queue.operations.add_job")
    @patch.object(CodeRepoFetcher, "fetch")
    def test_fetch_with_add_to_queue_failure(
        self, mock_fetch: Mock, mock_add_job: Mock, fetcher: CodeRepoFetcher
    ) -> None:
        """Test fetch_with_add_to_queue on failed fetch does not add job."""
        mock_fetch.return_value = FetchResult(
            url="https://github.com/user/repo",
            content_type="code_repo",
            success=False,
            error_message="Failed",
        )

        result = fetcher.fetch_with_add_to_queue("https://github.com/user/repo")

        assert result.success is False
        mock_add_job.assert_not_called()


class TestRepoMetadata:
    """Tests for RepoMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test RepoMetadata default values."""
        metadata = RepoMetadata()
        assert metadata.stars is None
        assert metadata.language is None
        assert metadata.description is None
        assert metadata.topics is None
        assert metadata.license is None
        assert metadata.fork_count is None
        assert metadata.open_issues is None
        assert metadata.default_branch is None
        assert metadata.repo_structure is None

    def test_custom_values(self) -> None:
        """Test RepoMetadata with custom values."""
        metadata = RepoMetadata(
            stars=100,
            language="Python",
            description="Test repo",
            topics=["python", "testing"],
            license="MIT",
            fork_count=50,
            open_issues=10,
            default_branch="main",
            repo_structure=["src/", "tests/"],
        )
        assert metadata.stars == 100
        assert metadata.language == "Python"
        assert metadata.description == "Test repo"
        assert metadata.topics == ["python", "testing"]
        assert metadata.license == "MIT"
        assert metadata.fork_count == 50
        assert metadata.open_issues == 10
        assert metadata.default_branch == "main"
        assert metadata.repo_structure == ["src/", "tests/"]
