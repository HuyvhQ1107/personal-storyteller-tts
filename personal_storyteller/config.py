"""Environment-driven configuration for the storyteller pipeline.

All settings are loaded from environment variables (or a local ``.env`` file
loaded by ``python-dotenv``).  Sensible defaults are baked in so that the
project works out of the box; copy ``.env.example`` to ``.env`` to override
them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()  # silently ignore if .env is missing
except ImportError:  # pragma: no cover - dotenv is in requirements.txt
    load_dotenv = None  # type: ignore[assignment]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# The prompt template the original notebook uses to steer the LLM.  We keep it
# here so it can be tested without instantiating the model.
STORY_PROMPT_TEMPLATE = (
    "Write an engaging and educational story about {topic} for beginners.\n"
    "Use simple and clear language to explain basic concepts.\n"
    "Include interesting facts and keep it friendly and encouraging.\n"
    "The story should be around {word_min}-{word_max} words and end with a "
    "brief summary of what we learned.\n"
    "Make it perfect for someone just starting to learn about this topic."
)


@dataclass(frozen=True)
class Settings:
    """Resolved configuration values.

    Frozen so callers cannot accidentally mutate shared state.  Use
    :func:`get_settings` (which honours environment variables) rather than
    instantiating this class directly when you want runtime overrides.
    """

    # LLM --------------------------------------------------------------
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 500
    do_sample: bool = False
    hf_token: str | None = None

    # Story prompt ------------------------------------------------------
    prompt_word_min: int = 200
    prompt_word_max: int = 300

    # TTS ---------------------------------------------------------------
    tts_lang: str = "en"
    tts_slow: bool = False

    # I/O ---------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def prompt(self, topic: str) -> str:
        """Build the generation prompt for ``topic`` using the resolved settings."""
        return STORY_PROMPT_TEMPLATE.format(
            topic=topic,
            word_min=self.prompt_word_min,
            word_max=self.prompt_word_max,
        )


def get_settings() -> Settings:
    """Return a :class:`Settings` instance built from environment variables."""
    out_dir = Path(os.getenv("STORYTELLER_OUTPUT_DIR", "output")).expanduser()
    return Settings(
        model_id=os.getenv("STORYTELLER_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
        max_new_tokens=_env_int("STORYTELLER_MAX_NEW_TOKENS", 500),
        do_sample=_env_bool("STORYTELLER_DO_SAMPLE", False),
        hf_token=os.getenv("HF_TOKEN") or None,
        tts_lang=os.getenv("STORYTELLER_TTS_LANG", "en"),
        tts_slow=_env_bool("STORYTELLER_TTS_SLOW", False),
        output_dir=out_dir,
    )


def as_dict(settings: Settings) -> dict[str, Any]:
    """Render settings as a plain dict (handy for logging / debugging)."""
    return {
        "model_id": settings.model_id,
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
        "tts_lang": settings.tts_lang,
        "tts_slow": settings.tts_slow,
        "output_dir": str(settings.output_dir),
        "hf_token_set": bool(settings.hf_token),
    }