"""Tests for personal_knowledge_base.interface.tutorial."""

from __future__ import annotations

import pytest

from personal_knowledge_base.interface.tutorial import (
    TutorialConfig,
    TutorialMode,
    TutorialSession,
    TutorialStep,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tm() -> TutorialMode:
    return TutorialMode()


@pytest.fixture()
def config_small() -> TutorialConfig:
    return TutorialConfig(max_step_chars=20)


# ---------------------------------------------------------------------------
# TutorialConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = TutorialConfig()
    assert cfg.max_step_chars == 500
    assert cfg.code_fence == "```"


def test_config_custom() -> None:
    cfg = TutorialConfig(max_step_chars=100, code_fence="~~~")
    assert cfg.max_step_chars == 100
    assert cfg.code_fence == "~~~"


# ---------------------------------------------------------------------------
# TutorialStep dataclass
# ---------------------------------------------------------------------------


def test_step_fields() -> None:
    step = TutorialStep(index=2, total=5, content="Hello", has_code=False, is_last=False)
    assert step.index == 2
    assert step.total == 5
    assert step.content == "Hello"
    assert not step.has_code
    assert not step.is_last


def test_step_last_flag() -> None:
    step = TutorialStep(index=3, total=3, content="Bye", has_code=False, is_last=True)
    assert step.is_last


# ---------------------------------------------------------------------------
# TutorialSession properties
# ---------------------------------------------------------------------------


def _make_session(n: int = 3) -> TutorialSession:
    steps = [
        TutorialStep(index=i, total=n, content=f"Step {i}", has_code=False, is_last=(i == n))
        for i in range(1, n + 1)
    ]
    return TutorialSession(steps=steps)


def test_session_current_step_initial() -> None:
    session = _make_session(3)
    assert session.current_step is not None
    assert session.current_step.index == 1


def test_session_current_step_after_advance() -> None:
    session = _make_session(3)
    session.advance()
    assert session.current_step is not None
    assert session.current_step.index == 2


def test_session_advance_returns_next_step() -> None:
    session = _make_session(3)
    next_step = session.advance()
    assert next_step is not None
    assert next_step.index == 2


def test_session_advance_past_end_returns_none() -> None:
    session = _make_session(2)
    session.advance()
    result = session.advance()
    assert result is None


def test_session_is_complete_false_initially() -> None:
    session = _make_session(3)
    assert not session.is_complete


def test_session_is_complete_true_after_last() -> None:
    session = _make_session(2)
    session.advance()
    session.advance()
    assert session.is_complete


def test_session_current_step_none_when_complete() -> None:
    session = _make_session(1)
    session.advance()
    assert session.current_step is None


def test_session_title() -> None:
    session = TutorialSession(steps=[], title="My Tutorial")
    assert session.title == "My Tutorial"


# ---------------------------------------------------------------------------
# Splitting: numbered headers
# ---------------------------------------------------------------------------

NUMBERED_HEADER_CONTENT = """\
## 1. Introduction

Welcome to the tutorial.

## 2. Installation

Run pip install.

## 3. Usage

Call the function.
"""


def test_split_numbered_headers(tm: TutorialMode) -> None:
    session = tm.create_session(NUMBERED_HEADER_CONTENT, title="Guide")
    assert len(session.steps) == 3
    assert session.title == "Guide"


def test_split_numbered_headers_indices(tm: TutorialMode) -> None:
    session = tm.create_session(NUMBERED_HEADER_CONTENT)
    for i, step in enumerate(session.steps, start=1):
        assert step.index == i
        assert step.total == 3


def test_split_numbered_headers_last_flag(tm: TutorialMode) -> None:
    session = tm.create_session(NUMBERED_HEADER_CONTENT)
    assert not session.steps[0].is_last
    assert not session.steps[1].is_last
    assert session.steps[2].is_last


def test_split_step_header(tm: TutorialMode) -> None:
    content = "### Step 1: First\n\nHello\n\n### Step 2: Second\n\nWorld"
    session = tm.create_session(content)
    assert len(session.steps) == 2


def test_split_hash_step_dash(tm: TutorialMode) -> None:
    content = "# Step 1 - Alpha\n\nA\n\n# Step 2 - Beta\n\nB"
    session = tm.create_session(content)
    assert len(session.steps) == 2


# ---------------------------------------------------------------------------
# Splitting: paragraph breaks
# ---------------------------------------------------------------------------


def test_split_paragraph_breaks(tm: TutorialMode) -> None:
    content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    session = tm.create_session(content)
    assert len(session.steps) == 3
    assert session.steps[0].content == "First paragraph."
    assert session.steps[1].content == "Second paragraph."
    assert session.steps[2].content == "Third paragraph."


def test_split_paragraph_drops_empty(tm: TutorialMode) -> None:
    content = "A\n\n\n\nB"
    session = tm.create_session(content)
    assert len(session.steps) == 2


# ---------------------------------------------------------------------------
# Splitting: hard char limit
# ---------------------------------------------------------------------------


def test_split_by_chars() -> None:
    cfg = TutorialConfig(max_step_chars=10)
    tm = TutorialMode(cfg)
    content = "A" * 25  # 25 chars → chunks of 10, 10, 5
    session = tm.create_session(content)
    assert len(session.steps) == 3
    assert len(session.steps[0].content) == 10
    assert len(session.steps[1].content) == 10
    assert len(session.steps[2].content) == 5


def test_split_by_chars_exact_boundary() -> None:
    cfg = TutorialConfig(max_step_chars=10)
    tm = TutorialMode(cfg)
    content = "B" * 20
    session = tm.create_session(content)
    assert len(session.steps) == 2


def test_split_single_chunk_no_breaks() -> None:
    tm = TutorialMode(TutorialConfig(max_step_chars=1000))
    content = "Just one block of text with no headers or paragraph breaks."
    session = tm.create_session(content)
    assert len(session.steps) == 1
    assert session.steps[0].content == content


# ---------------------------------------------------------------------------
# Code block detection
# ---------------------------------------------------------------------------


def test_has_code_true(tm: TutorialMode) -> None:
    content = "## 1. Example\n\n```python\nprint('hi')\n```\n\n## 2. End\n\nDone."
    session = tm.create_session(content)
    assert session.steps[0].has_code is True
    assert session.steps[1].has_code is False


def test_has_code_paragraph_split(tm: TutorialMode) -> None:
    content = "Intro text.\n\n```bash\necho hello\n```"
    session = tm.create_session(content)
    code_step = next(s for s in session.steps if s.has_code)
    assert code_step.has_code is True


def test_has_code_false_when_no_fence(tm: TutorialMode) -> None:
    content = "No code here.\n\nJust text."
    session = tm.create_session(content)
    for step in session.steps:
        assert step.has_code is False


def test_custom_code_fence() -> None:
    cfg = TutorialConfig(code_fence="~~~")
    tm = TutorialMode(cfg)
    content = "Step one.\n\n~~~python\nx = 1\n~~~"
    session = tm.create_session(content)
    assert any(s.has_code for s in session.steps)


# ---------------------------------------------------------------------------
# format_step
# ---------------------------------------------------------------------------


def test_format_step_non_last(tm: TutorialMode) -> None:
    step = TutorialStep(index=1, total=3, content="Hello world", has_code=False, is_last=False)
    result = tm.format_step(step)
    assert result.startswith("Step 1/3")
    assert "Hello world" in result
    assert "[Next: type 'next' to continue]" in result
    assert "[End of tutorial]" not in result


def test_format_step_last(tm: TutorialMode) -> None:
    step = TutorialStep(index=3, total=3, content="Goodbye", has_code=False, is_last=True)
    result = tm.format_step(step)
    assert "Step 3/3" in result
    assert "Goodbye" in result
    assert "[End of tutorial]" in result
    assert "[Next:" not in result


def test_format_step_structure(tm: TutorialMode) -> None:
    step = TutorialStep(index=2, total=4, content="Middle", has_code=False, is_last=False)
    result = tm.format_step(step)
    parts = result.split("\n\n")
    assert parts[0] == "Step 2/4"
    assert parts[1] == "Middle"
    assert parts[2] == "[Next: type 'next' to continue]"


# ---------------------------------------------------------------------------
# process_command
# ---------------------------------------------------------------------------


def _make_tm_session(content: str = "") -> tuple[TutorialMode, TutorialSession]:
    if not content:
        content = "First.\n\nSecond.\n\nThird."
    tm = TutorialMode()
    session = tm.create_session(content)
    return tm, session


def test_command_next_advances(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.\n\nC.")
    tm.process_command(session, "next")
    assert session.current_index == 1


def test_command_next_returns_formatted(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "next")
    assert "Step 2/2" in result


def test_command_continue(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "continue")
    assert "Step 2/2" in result


def test_command_next_at_end_returns_ended(tm: TutorialMode) -> None:
    session = tm.create_session("Only one step here.")
    result = tm.process_command(session, "next")
    assert result == "Tutorial ended."


def test_command_back(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.\n\nC.")
    session.current_index = 2
    result = tm.process_command(session, "back")
    assert "Step 2/3" in result
    assert session.current_index == 1


def test_command_previous(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    session.current_index = 1
    result = tm.process_command(session, "previous")
    assert "Step 1/2" in result


def test_command_back_at_start_stays_at_first(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "back")
    assert "Step 1/2" in result
    assert session.current_index == 0


def test_command_restart(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.\n\nC.")
    session.current_index = 2
    result = tm.process_command(session, "restart")
    assert session.current_index == 0
    assert "Step 1/3" in result


def test_command_quit(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    assert tm.process_command(session, "quit") == "Tutorial ended."


def test_command_exit(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    assert tm.process_command(session, "exit") == "Tutorial ended."


def test_command_stop(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    assert tm.process_command(session, "stop") == "Tutorial ended."


def test_command_unknown_shows_hint(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "skip")
    assert "Unknown command 'skip'" in result
    assert "Step 1/2" in result  # current step still shown


def test_command_unknown_shows_valid_commands(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "hello")
    assert "next" in result
    assert "quit" in result


def test_command_case_insensitive(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "NEXT")
    assert "Step 2/2" in result


def test_command_whitespace_stripped(tm: TutorialMode) -> None:
    session = tm.create_session("A.\n\nB.")
    result = tm.process_command(session, "  next  ")
    assert "Step 2/2" in result


# ---------------------------------------------------------------------------
# TutorialMode default config
# ---------------------------------------------------------------------------


def test_default_config_used_when_none() -> None:
    tm = TutorialMode(None)
    assert tm.config.max_step_chars == 500
    assert tm.config.code_fence == "```"


def test_custom_config_passed_through() -> None:
    cfg = TutorialConfig(max_step_chars=50)
    tm = TutorialMode(cfg)
    assert tm.config.max_step_chars == 50


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_content_produces_no_steps(tm: TutorialMode) -> None:
    session = tm.create_session("")
    assert len(session.steps) == 0
    assert session.current_step is None
    assert session.is_complete


def test_whitespace_only_content(tm: TutorialMode) -> None:
    session = tm.create_session("   \n\n   ")
    assert len(session.steps) == 0


def test_single_step_is_last(tm: TutorialMode) -> None:
    session = tm.create_session("Just one step.")
    assert len(session.steps) == 1
    assert session.steps[0].is_last is True
    assert session.steps[0].index == 1
    assert session.steps[0].total == 1
