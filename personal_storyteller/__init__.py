"""Use Mistral and gTTS to Create Your Personal Storyteller.

A small library + CLI that turns a topic into a short, spoken-word story:

    topic  ->  local LLM  ->  story  ->  gTTS  ->  MP3  ->  speakers
"""

from __future__ import annotations

from .config import Settings, get_settings
from .player import play_bytes, play_file
from .storyteller import GenerationParams, StoryGenerator
from .tts import TTS, save_bytes, slugify_topic

__version__ = "0.1.0"

__all__ = [
    "GenerationParams",
    "Settings",
    "StoryGenerator",
    "TTS",
    "get_settings",
    "play_bytes",
    "play_file",
    "save_bytes",
    "slugify_topic",
    "__version__",
]