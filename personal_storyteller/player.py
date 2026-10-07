"""Audio playback.

The original lab uses the ``IPython.display.Audio`` widget to play audio
inline in a Jupyter notebook.  In a stand-alone Python project we instead
hand the bytes to ``pygame.mixer`` (a cross-platform, zero-dependency
choice) and block until the sound finishes.

Playback is opt-in: the CLI only calls into here when ``--play`` is given,
and the Python API treats each function as a no-op when the user passes
``play=False`` (the default in :mod:`personal_storyteller.cli`).
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def play_bytes(audio: bytes, *, autoplay: bool = True) -> None:
    """Play raw MP3/WAV audio bytes.

    Pygame's MP3 support depends on the system having an MP3 codec available;
    when that fails we transparently fall back to writing the bytes to a
    temporary file and asking the OS to open it with the default player.
    """
    if not audio:
        raise ValueError("play_bytes() needs a non-empty bytes object.")

    if not autoplay:
        return

    try:
        import pygame  # type: ignore[import-not-found]

        pygame.mixer.init()
        try:
            sound = pygame.mixer.Sound(buffer=audio)
        except Exception:  # pragma: no cover - depends on platform codecs
            logger.warning(
                "pygame could not decode MP3 in memory (codec missing). "
                "Falling back to the OS default player."
            )
            _play_with_default_player(audio)
            return

        channel = sound.play()
        if channel is not None:
            while channel.get_busy():
                pygame.time.wait(50)
    except ImportError:  # pragma: no cover - pygame is in requirements.txt
        logger.warning(
            "pygame is not installed; falling back to the OS default player."
        )
        _play_with_default_player(audio)


def play_file(path: PathLike, *, autoplay: bool = True) -> None:
    """Play an audio file from disk."""
    if not autoplay:
        return
    if not Path(path).exists():
        raise FileNotFoundError(path)

    try:
        import pygame  # type: ignore[import-not-found]

        pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    except ImportError:  # pragma: no cover
        _play_with_default_player(Path(path).read_bytes())


def _play_with_default_player(audio: bytes) -> None:
    """Last-resort playback: hand the audio to the OS default player."""
    import os
    import subprocess
    import sys

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio)
        tmp_path = tmp.name

    try:
        if sys.platform.startswith("win"):
            os.startfile(tmp_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", tmp_path])
        else:
            subprocess.Popen(["xdg-open", tmp_path])
    finally:
        # Leave the file behind; the OS will clean up the temp dir eventually.
        # (Trying to delete it immediately would race with the player.)
        logger.debug("Wrote temp audio to %s", tmp_path)


__all__ = ["play_bytes", "play_file"]


# Silence unused-var warning for io, kept for type re-export consistency.
_ = io