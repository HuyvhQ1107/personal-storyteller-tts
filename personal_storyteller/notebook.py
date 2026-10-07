"""Notebook-style examples for interactive / Jupyter use.

These mirror the cells of the original
``Use_Mistral_and_gTTS_to_Create_Your_Personal_Storyteller.ipynb`` notebook
but in plain Python so they can be imported from anywhere.

The functions in here *do not* actually load the LLM or hit Google's TTS API
during import — call them explicitly.  This makes the module safe to import
inside unit tests.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .storyteller import GenerationParams, StoryGenerator
from .tts import TTS


def make_generator(model_id: str | None = None) -> StoryGenerator:
    """Helper that loads the default StoryGenerator (mirrors cell 7)."""
    return StoryGenerator.from_defaults(model_id=model_id)


def synthesize_and_save(story: str, output_path: str, lang: str = "en") -> str:
    """Run ``gTTS`` on ``story`` and save it to ``output_path`` (mirrors cell 12)."""
    TTS(lang=lang).synthesize_to_file(story, output_path)
    return output_path


def audio_widget_for(story: str, lang: str = "en") -> Any:
    """Return an IPython Audio widget for the synthesized bytes (mirrors cell 10).

    Requires IPython to be installed (``pip install ipython ipywidgets``).
    Raises :class:`ImportError` with a helpful message otherwise.
    """
    try:
        from IPython.display import Audio
    except ImportError as exc:  # pragma: no cover - ipython is optional
        raise ImportError(
            "audio_widget_for() needs IPython. Install it with "
            "`pip install ipython ipywidgets`."
        ) from exc

    import io

    audio_bytes = TTS(lang=lang).synthesize_bytes(story)
    return Audio(io.BytesIO(audio_bytes).read(), autoplay=False)


__all__ = [
    "GenerationParams",
    "Settings",
    "StoryGenerator",
    "TTS",
    "audio_widget_for",
    "make_generator",
    "synthesize_and_save",
]