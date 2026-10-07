"""Text-to-speech helpers built on top of ``gTTS``.

Two flavours are exposed:

* :class:`TTS` – an object-oriented wrapper that mirrors the notebook's
  ``gTTS(story)`` -> ``tts.write_to_fp(buffer)`` pattern.
* :func:`save_bytes` – a free-standing convenience for writing pre-encoded
  MP3 bytes to disk.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Union


PathLike = Union[str, Path]


@dataclass
class TTS:
    """Lazy-initialised wrapper around :class:`gtts.gTTS`.

    The underlying ``gTTS`` instance is built on the first call to
    :meth:`synthesize_bytes` / :meth:`synthesize_file` so constructing a
    ``TTS`` is essentially free and cheap to share.
    """

    lang: str = "en"
    slow: bool = False
    tld: str = "com"

    # ----------------------------------------------------------------- private
    def _build(self, text: str) -> "gtts.gTTS":  # type: ignore[name-defined]
        # Imported lazily so the package import cost stays minimal.
        from gtts import gTTS

        return gTTS(text=text, lang=self.lang, slow=self.slow, tld=self.tld)

    # ------------------------------------------------------------------ public
    def synthesize_bytes(self, text: str) -> bytes:
        """Return the synthesized speech as raw MP3 bytes (in-memory)."""
        buffer = io.BytesIO()
        self._build(text).write_to_fp(buffer)
        return buffer.getvalue()

    def synthesize_to_file(self, text: str, path: PathLike) -> Path:
        """Synthesize and save directly to ``path`` (skips the in-memory copy)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._build(text).save(str(out))
        return out

    def write_to_fp(self, text: str, fp: IO[bytes]) -> int:
        """Write the synthesized audio to an already-open binary file-like object."""
        self._build(text).write_to_fp(fp)
        return fp.tell()


def save_bytes(audio: bytes, path: PathLike) -> Path:
    """Save raw MP3 ``audio`` to ``path`` (creating parent directories)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(audio)
    return out


def slugify_topic(topic: str, max_words: int = 6) -> str:
    """Make a topic safe to embed in a filename."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_ " else "_" for ch in topic.strip().lower()
    )
    parts = cleaned.split()
    return "_".join(parts[:max_words]) or "story"


__all__ = ["TTS", "PathLike", "save_bytes", "slugify_topic"]