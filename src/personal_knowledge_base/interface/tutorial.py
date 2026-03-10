"""Tutorial mode for the Personal Knowledge Base.

Breaks tutorial content into sequential steps and presents them one at a time.
Supports numbered markdown headers, paragraph breaks, and hard-size splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TutorialConfig:
    """Configuration for tutorial mode.

    Attributes:
        max_step_chars: Maximum characters allowed per step before hard-splitting.
        code_fence: Delimiter used to detect code blocks.

    """

    max_step_chars: int = 500
    code_fence: str = "```"


@dataclass
class TutorialStep:
    """A single step within a tutorial.

    Attributes:
        index: 1-based step number.
        total: Total number of steps in the session.
        content: The step's text content.
        has_code: True if the step contains a fenced code block.
        is_last: True if this is the final step.

    """

    index: int
    total: int
    content: str
    has_code: bool
    is_last: bool


@dataclass
class TutorialSession:
    """Active tutorial session tracking progress through steps.

    Attributes:
        steps: Ordered list of all steps.
        current_index: 0-based index of the step currently being viewed.
        title: Optional title for the tutorial.

    """

    steps: list[TutorialStep]
    current_index: int = 0
    title: str = ""

    @property
    def current_step(self) -> TutorialStep | None:
        """Return the step at *current_index*, or None if out of bounds."""
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        """True when the user has advanced past the last step."""
        return self.current_index >= len(self.steps)

    def advance(self) -> TutorialStep | None:
        """Move to the next step.

        Returns:
            The next TutorialStep, or None if the session is already complete.

        """
        self.current_index += 1
        return self.current_step


# ---------------------------------------------------------------------------
# Regex patterns for numbered-header splitting
# ---------------------------------------------------------------------------
# Matches lines like: "## 1.", "### Step 1:", "# Step 1 -", "## Step 2 –"
_NUMBERED_HEADER_RE = re.compile(
    r"^#{1,6}\s+(?:step\s+)?\d+\s*[.:\-–]",
    re.IGNORECASE | re.MULTILINE,
)


class TutorialMode:
    """Parse tutorial content and manage step-by-step presentation.

    Example::

        tm = TutorialMode()
        session = tm.create_session("## 1. Intro\\n\\nHello\\n\\n## 2. Body\\n\\nWorld")
        print(tm.format_step(session.current_step))
        tm.process_command(session, "next")

    """

    def __init__(self, config: TutorialConfig | None = None) -> None:
        """Initialise with optional config.

        Args:
            config: Configuration dataclass. Defaults to TutorialConfig().

        """
        self.config = config if config is not None else TutorialConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, content: str, title: str = "") -> TutorialSession:
        """Parse *content* into a TutorialSession with ordered steps.

        Splitting priority:
        1. Numbered markdown headers (``## 1.``, ``### Step 1:``, etc.)
        2. Double newlines — only when result has ≥ 2 non-empty paragraphs.
        3. Hard split by ``max_step_chars`` when content is one long block.

        Args:
            content: Raw tutorial text.
            title: Optional title for the session.

        Returns:
            A TutorialSession ready to use.

        """
        raw_steps = self._split_content(content)
        steps = self._build_steps(raw_steps)
        return TutorialSession(steps=steps, current_index=0, title=title)

    def format_step(self, step: TutorialStep) -> str:
        """Format a step for display.

        Args:
            step: The step to format.

        Returns:
            A formatted string with header, content, and navigation hint.

        """
        header = f"Step {step.index}/{step.total}"
        footer = "[End of tutorial]" if step.is_last else "[Next: type 'next' to continue]"
        return f"{header}\n\n{step.content}\n\n{footer}"

    def process_command(self, session: TutorialSession, command: str) -> str:
        """Handle a user command within an active tutorial session.

        Supported commands:
        - ``next`` / ``continue`` → advance to next step
        - ``back`` / ``previous`` → go back one step
        - ``restart`` → reset to step 1
        - ``quit`` / ``exit`` / ``stop`` → end the tutorial
        - anything else → re-display the current step with a hint

        Args:
            session: The active TutorialSession.
            command: Raw command string from the user.

        Returns:
            Formatted response string.

        """
        cmd = command.strip().lower()

        if cmd in ("quit", "exit", "stop"):
            return "Tutorial ended."

        if cmd in ("next", "continue"):
            next_step = session.advance()
            if next_step is None:
                return "Tutorial ended."
            return self.format_step(next_step)

        if cmd in ("back", "previous"):
            if session.current_index > 0:
                session.current_index -= 1
            step = session.current_step
            if step is None:
                return "Tutorial ended."
            return self.format_step(step)

        if cmd == "restart":
            session.current_index = 0
            step = session.current_step
            if step is None:
                return "Tutorial ended."
            return self.format_step(step)

        # Unknown command — show current step again with a hint
        step = session.current_step
        if step is None:
            return "Tutorial ended."
        hint = f"Unknown command '{command}'. " "Use 'next', 'back', 'restart', or 'quit'.\n\n"
        return hint + self.format_step(step)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_content(self, content: str) -> list[str]:
        """Return a list of raw step strings from *content*.

        Applies splitting strategies in priority order.
        """
        # Strategy 1: numbered markdown headers
        parts = self._split_by_numbered_headers(content)
        if len(parts) >= 2:
            return parts

        # Strategy 2: double newlines (paragraph breaks)
        parts = self._split_by_paragraphs(content)
        if len(parts) >= 2:
            return parts

        # Strategy 3: hard split by max_step_chars
        return self._split_by_chars(content)

    def _split_by_numbered_headers(self, content: str) -> list[str]:
        """Split on numbered markdown headers, returning non-empty parts."""
        matches = list(_NUMBERED_HEADER_RE.finditer(content))
        if not matches:
            return []

        parts: list[str] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            chunk = content[start:end].strip()
            if chunk:
                parts.append(chunk)
        return parts

    def _split_by_paragraphs(self, content: str) -> list[str]:
        """Split on double newlines, keeping non-empty paragraphs."""
        raw = re.split(r"\n\s*\n", content)
        return [p.strip() for p in raw if p.strip()]

    def _split_by_chars(self, content: str) -> list[str]:
        """Hard-split *content* into chunks of at most *max_step_chars*."""
        text = content.strip()
        if not text:
            return []
        size = self.config.max_step_chars
        return [text[i : i + size] for i in range(0, len(text), size)]

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_steps(self, raw_steps: list[str]) -> list[TutorialStep]:
        """Convert raw string chunks into TutorialStep objects."""
        total = len(raw_steps)
        steps: list[TutorialStep] = []
        for idx, chunk in enumerate(raw_steps, start=1):
            steps.append(
                TutorialStep(
                    index=idx,
                    total=total,
                    content=chunk,
                    has_code=self.config.code_fence in chunk,
                    is_last=(idx == total),
                )
            )
        return steps
