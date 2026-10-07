"""Smoke tests for the storyteller pipeline.

These tests deliberately avoid loading the local LLM (which would download
several GB of weights and require a GPU/CPU with significant capacity).
The full behaviour is verified by the demo script in ``examples/``.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from personal_storyteller import (
    GenerationParams,
    Settings,
    StoryGenerator,
    TTS,
    play_bytes,
    save_bytes,
    slugify_topic,
)
from personal_storyteller.config import STORY_PROMPT_TEMPLATE, get_settings


# ---------------------------------------------------------------------- config
def test_settings_default_model():
    cfg = get_settings()
    assert cfg.model_id  # non-empty string
    assert cfg.max_new_tokens > 0


def test_settings_prompt_contains_topic():
    cfg = Settings(model_id="m", max_new_tokens=10, do_sample=False)
    prompt = cfg.prompt("the life cycle of butterflies")
    assert "the life cycle of butterflies" in prompt
    assert "200" in prompt and "300" in prompt


def test_prompt_template_format_invariants():
    rendered = STORY_PROMPT_TEMPLATE.format(
        topic="x", word_min=10, word_max=20
    )
    assert "{topic}" not in rendered
    assert "{word_min}" not in rendered
    assert "beginners" in rendered


# ------------------------------------------------------------------ storyteller
class _FakePipeline:
    """Stub that mimics ``transformers.pipeline("text-generation", ...)``."""

    def __init__(self, reply: str):
        self._reply = reply
        self.last_kwargs: dict | None = None
        self.last_messages: list | None = None

    def __call__(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return [{"generated_text": messages + [{"role": "assistant", "content": self._reply}]}]


def test_generate_uses_chat_messages_and_returns_assistant_text():
    pipe = _FakePipeline("Once upon a time…")
    gen = StoryGenerator(pipe, params=GenerationParams(max_new_tokens=42))

    story = gen.generate("the moon")

    assert story == "Once upon a time…"
    assert pipe.last_kwargs == {
        "max_new_tokens": 42,
        "do_sample": False,
    }
    assert pipe.last_messages[0]["role"] == "user"
    assert "the moon" in pipe.last_messages[0]["content"]


def test_generate_streamed_joins_chunks():
    """Stream helper falls back to a single chunk when no streamer is available."""
    # When transformers isn't installed, stream() falls back to a single yield
    # of the non-streaming generate() result.
    fake = _FakePipeline("Streamed once.")
    fake.tokenizer = None  # no tokenizer -> force the fallback path
    gen = StoryGenerator(fake, params=GenerationParams(max_new_tokens=10))
    chunks = list(gen.stream("the moon"))
    assert "".join(chunks).strip() == "Streamed once."


# ---------------------------------------------------------------------------- tts
def test_tts_synthesize_bytes(monkeypatch):
    """Synthesize without hitting the network: patch ``gTTS`` itself."""

    class _FakeGTTS:
        def __init__(self, text, lang="en", slow=False, tld="com"):
            self.text = text
            self.lang = lang
            self.slow = slow
            self.tld = tld

        def write_to_fp(self, fp):
            fp.write(b"FAKE-MP3:" + self.text.encode())

        def save(self, path):
            Path(path).write_bytes(b"FAKE-MP3:" + self.text.encode())

    import sys
    monkeypatch.setitem(sys.modules, "gtts", MagicMock(gTTS=_FakeGTTS))

    audio = TTS().synthesize_bytes("hello")
    assert audio == b"FAKE-MP3:hello"


def test_save_bytes_writes_file(tmp_path: Path):
    target = tmp_path / "sub" / "story.mp3"
    out = save_bytes(b"abc", target)
    assert out == target
    assert target.read_bytes() == b"abc"


def test_slugify_topic_normalises():
    assert slugify_topic("The Life Cycle of Butterflies!") == "the_life_cycle_of_butterflies_"
    assert slugify_topic("   ") == "story"


# ---------------------------------------------------------------------- player
def test_play_bytes_rejects_empty():
    with pytest.raises(ValueError):
        play_bytes(b"", autoplay=True)


def test_play_bytes_noop_when_disabled():
    # Should not raise even without pygame loaded.
    play_bytes(b"not-a-real-mp3", autoplay=False)


# -------------------------------------------------------------------- __all__
def test_public_api_exports():
    import personal_storyteller as pkg

    expected = {
        "GenerationParams",
        "Settings",
        "StoryGenerator",
        "TTS",
        "get_settings",
        "play_bytes",
        "play_file",
        "save_bytes",
        "__version__",
    }
    assert expected.issubset(set(pkg.__all__))


def test_cli_help_runs():
    from click.testing import CliRunner

    from personal_storyteller.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "TOPIC" in result.output


def test_cli_show_config_uses_settings():
    from click.testing import CliRunner

    from personal_storyteller.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["anything", "--show-config"])
    assert result.exit_code == 0
    assert "model_id" in result.output