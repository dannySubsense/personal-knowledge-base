"""Code repository fetcher for GitHub and other git repos."""

import contextlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult


@dataclass
class RepoMetadata:
    """Metadata extracted from a code repository.

    Attributes:
        stars: Number of stars (GitHub only).
        language: Primary programming language.
        description: Repository description.
        topics: List of topics/tags.
        license: License name.
        fork_count: Number of forks (GitHub only).
        open_issues: Number of open issues (GitHub only).
        default_branch: Default branch name.
        repo_structure: Basic repository structure (top-level directories and files).

    """

    stars: int | None = None
    language: str | None = None
    description: str | None = None
    topics: list[str] | None = None
    license: str | None = None
    fork_count: int | None = None
    open_issues: int | None = None
    default_branch: str | None = None
    repo_structure: list[str] | None = None


class CodeRepoFetcher(Fetcher):
    """Fetcher for code repositories from GitHub and other git hosts.

    Clones repositories to a temporary directory, extracts README content,
    and fetches metadata from the GitHub API when available.

    Supported URL formats:
        - https://github.com/user/repo
        - https://github.com/user/repo.git
        - https://gitlab.com/user/repo
        - https://bitbucket.org/user/repo
        - git@github.com:user/repo.git

    """

    # Regex patterns for extracting repo info from URLs
    _GITHUB_URL_PATTERN = re.compile(
        r"^(?:https?://|git@)github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$"
    )

    # README file names to look for (in order of preference)
    _README_NAMES = ["README.md", "README.rst", "README.txt", "README"]

    def __init__(self, github_token: str | None = None) -> None:
        """Initialize the code repository fetcher.

        Args:
            github_token: Optional GitHub personal access token for API access.
                         Increases rate limits and allows access to private repos.

        """
        self.github_token = github_token
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Get or create the requests session.

        Returns:
            requests.Session instance.

        """
        if self._session is None:
            self._session = requests.Session()
            if self.github_token:
                self._session.headers["Authorization"] = f"token {self.github_token}"
        return self._session

    def can_fetch(self, url: str) -> bool:
        """Check if this fetcher can handle the given URL.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a valid git repository URL, False otherwise.

        """
        parsed = urlparse(url)

        # Check for common git hosts
        if parsed.netloc in ("github.com", "www.github.com", "gitlab.com", "bitbucket.org"):
            return True

        # Check for git@ format
        if url.startswith("git@"):
            return True

        # Check for .git suffix
        return bool(url.endswith(".git"))

    def _parse_github_repo(self, url: str) -> tuple[str, str] | None:
        """Parse owner and repo name from a GitHub URL.

        Args:
            url: The GitHub repository URL.

        Returns:
            Tuple of (owner, repo_name) or None if not a valid GitHub URL.

        """
        match = self._GITHUB_URL_PATTERN.match(url)
        if match:
            return match.group(1), match.group(2)
        return None

    def _fetch_github_metadata(self, owner: str, repo: str) -> RepoMetadata | None:
        """Fetch metadata from the GitHub API.

        Args:
            owner: Repository owner/organization.
            repo: Repository name.

        Returns:
            RepoMetadata object or None if fetch failed.

        """
        session = self._get_session()
        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        try:
            response = session.get(api_url, timeout=30)

            if response.status_code == 404:
                return None  # Repository not found
            if response.status_code == 403:
                # Rate limit or authentication issue - continue without metadata
                return None
            if not response.ok:
                return None

            data = response.json()

            return RepoMetadata(
                stars=data.get("stargazers_count"),
                language=data.get("language"),
                description=data.get("description"),
                topics=data.get("topics", []),
                license=data.get("license", {}).get("name") if data.get("license") else None,
                fork_count=data.get("forks_count"),
                open_issues=data.get("open_issues_count"),
                default_branch=data.get("default_branch"),
            )
        except (requests.RequestException, ValueError):
            return None

    def _clone_repo(self, url: str, target_dir: str) -> bool:
        """Clone a git repository to a target directory.

        Performs a shallow clone (--depth 1) to minimize download size.

        Args:
            url: The repository URL.
            target_dir: Directory to clone into.

        Returns:
            True if clone succeeded, False otherwise.

        """
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, target_dir],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _find_readme(self, repo_dir: str) -> Path | None:
        """Find the README file in the repository.

        Args:
            repo_dir: Path to the repository directory.

        Returns:
            Path to the README file or None if not found.

        """
        repo_path = Path(repo_dir)

        for readme_name in self._README_NAMES:
            readme_path = repo_path / readme_name
            if readme_path.exists() and readme_path.is_file():
                return readme_path

            # Check case-insensitive
            for item in repo_path.iterdir():
                if item.name.lower() == readme_name.lower() and item.is_file():
                    return item

        return None

    def _read_readme(self, readme_path: Path) -> str:
        """Read the contents of a README file.

        Args:
            readme_path: Path to the README file.

        Returns:
            Contents of the README file.

        """
        try:
            return readme_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _get_repo_structure(self, repo_dir: str, max_items: int = 50) -> list[str]:
        """Get the top-level structure of the repository.

        Args:
            repo_dir: Path to the repository directory.
            max_items: Maximum number of items to return.

        Returns:
            List of top-level file and directory names.

        """
        repo_path = Path(repo_dir)
        structure = []

        try:
            for item in sorted(repo_path.iterdir()):
                if item.name.startswith("."):
                    continue  # Skip hidden files
                if item.is_dir():
                    structure.append(f"{item.name}/")
                else:
                    structure.append(item.name)

                if len(structure) >= max_items:
                    break
        except OSError:
            pass

        return structure

    def _extract_title_from_readme(self, readme_content: str) -> str:
        """Extract the title from README content.

        Looks for the first heading (H1) or uses the first line.

        Args:
            readme_content: Content of the README file.

        Returns:
            Extracted title or empty string.

        """
        lines = readme_content.split("\n")

        for line in lines:
            line = line.strip()
            # Look for markdown H1
            if line.startswith("# "):
                return line[2:].strip()
            # Look for RST title (underlined with ===)
            if line and len(lines) > lines.index(line) + 1:
                next_line = lines[lines.index(line) + 1].strip()
                if next_line and all(c == "=" for c in next_line):
                    return line

        # Fallback to first non-empty line
        for line in lines:
            stripped = line.strip()
            if stripped:
                return stripped[:100]  # Limit length

        return ""

    def fetch(self, url: str) -> FetchResult:
        """Fetch content from a code repository.

        Args:
            url: The repository URL.

        Returns:
            FetchResult containing the README content and metadata.

        """
        if not self.can_fetch(url):
            return FetchResult(
                url=url,
                content_type="code_repo",
                success=False,
                error_message="Invalid repository URL: Not a recognized git repository URL.",
            )

        # Parse GitHub repo info for API metadata
        github_info = self._parse_github_repo(url)
        metadata: dict[str, Any] = {}
        repo_metadata: RepoMetadata | None = None

        if github_info:
            owner, repo = github_info
            repo_metadata = self._fetch_github_metadata(owner, repo)
            if repo_metadata is None:
                # Repository not found on GitHub
                return FetchResult(
                    url=url,
                    content_type="code_repo",
                    success=False,
                    error_message="Repository not found (404): The GitHub repository does not exist or is not accessible.",
                )

        # Create temporary directory for cloning
        temp_dir = tempfile.mkdtemp(prefix="repo_fetch_")

        try:
            # Clone the repository
            clone_success = self._clone_repo(url, temp_dir)

            if not clone_success:
                return FetchResult(
                    url=url,
                    content_type="code_repo",
                    success=False,
                    error_message="Failed to clone repository: The URL may not be a valid git repository or the repository is not accessible.",
                )

            # Find and read README
            readme_path = self._find_readme(temp_dir)

            if readme_path is None:
                return FetchResult(
                    url=url,
                    content_type="code_repo",
                    success=False,
                    error_message="No README found: The repository does not contain a README file.",
                )

            readme_content = self._read_readme(readme_path)

            if not readme_content.strip():
                return FetchResult(
                    url=url,
                    content_type="code_repo",
                    success=False,
                    error_message="README is empty: The README file exists but contains no content.",
                )

            # Extract title from README
            title = self._extract_title_from_readme(readme_content)

            # Get repository structure
            repo_structure = self._get_repo_structure(temp_dir)

            # Build metadata
            if repo_metadata:
                metadata = {
                    "stars": repo_metadata.stars,
                    "language": repo_metadata.language,
                    "description": repo_metadata.description,
                    "topics": repo_metadata.topics,
                    "license": repo_metadata.license,
                    "fork_count": repo_metadata.fork_count,
                    "open_issues": repo_metadata.open_issues,
                    "default_branch": repo_metadata.default_branch,
                    "repo_structure": repo_structure,
                    "github_owner": owner,
                    "github_repo": repo,
                }
            else:
                metadata = {
                    "repo_structure": repo_structure,
                }

            return FetchResult(
                url=url,
                title=title or "Code Repository",
                content=readme_content,
                content_type="code_repo",
                success=True,
                metadata=metadata,
            )

        finally:
            # Clean up temporary directory
            with contextlib.suppress(OSError):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def fetch_with_add_to_queue(
        self,
        url: str,
        priority: int = 2,
        kb_name: str | None = None,
    ) -> FetchResult:
        """Fetch repository and add to ingestion queue.

        This is a convenience method that fetches the repository and
        automatically adds it to the job queue for processing.

        Args:
            url: The repository URL.
            priority: Priority level (1=immediate, 2=normal).
            kb_name: Target knowledge base name.

        Returns:
            FetchResult containing the repository content.

        """
        from personal_knowledge_base.queue.operations import add_job

        result = self.fetch(url)

        if result.success:
            add_job(
                url=url,
                priority=priority,
                content_type="code_repo",
                kb_name=kb_name,
            )

        return result
