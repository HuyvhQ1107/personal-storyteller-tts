"""End-to-end demo for the storyteller pipeline.

Run::

    python examples\generate_demo.py

The script:

1. Loads the configured HuggingFace model (Qwen2.5-3B-Instruct by default).
2. Generates a short story about "the life cycle of butterflies".
3. Saves it to ``output/butterflies.mp3`` (relative to the repo root).
4. Plays it through pygame.

Pass ``--no-play`` to skip playback in headless environments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personal_storyteller import (  # noqa: E402  (import after sys.path tweak)
    Settings,
    StoryGenerator,
    TTS,
    play_bytes,
    save_bytes,
)
from personal_storyteller.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the storyteller end-to-end.")
    parser.add_argument(
        "--topic",
        default="the life cycle of butterflies",
        help="Topic for the generated story.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="MP3 output path (default: <output-dir>/<slug>.mp3).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip playback (useful for CI / headless runs).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the HuggingFace model id.",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Override the gTTS language code.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo] model id:  {settings.model_id}", file=sys.stderr)
    print(f"[demo] topic:    {args.topic!r}", file=sys.stderr)

    generator = StoryGenerator.from_defaults(
        model_id=args.model,
        settings=settings,
    )

    print("[demo] generating story…", file=sys.stderr)
    story = generator.generate(args.topic)
    print()
    print(story)
    print()

    tts = TTS(lang=args.lang or settings.tts_lang, slow=settings.tts_slow)
    audio = tts.synthesize_bytes(story)

    from personal_storyteller.tts import slugify_topic

    out_path = args.output or (out_dir / f"{slugify_topic(args.topic)}.mp3")
    save_bytes(audio, out_path)
    print(f"[demo] saved MP3 to {out_path}", file=sys.stderr)

    if not args.no_play:
        print("[demo] playing audio…", file=sys.stderr)
        play_bytes(audio)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())