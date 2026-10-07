"""Command-line entry point for the storyteller pipeline.

Usage examples::

    storyteller "the life cycle of butterflies" -o output\butterflies.mp3 --play
    storyteller "romantic relationships" --no-play
    storyteller "the discovery of penicillin" --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from .config import as_dict, get_settings
from .player import play_bytes
from .storyteller import StoryGenerator
from .tts import TTS, slugify_topic

logger = logging.getLogger(__name__)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Generate a spoken-word story for TOPIC.\n\n"
        "TOPIC is the subject you want a short educational story about "
        "(e.g. 'the life cycle of butterflies')."
    ),
)
@click.argument("topic")
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save the MP3. Defaults to <output-dir>/<slug>.mp3.",
)
@click.option(
    "--play/--no-play",
    default=True,
    help="Play the audio once synthesis is done (default: play).",
)
@click.option(
    "--save/--no-save",
    default=True,
    help="Save the MP3 to disk (default: save).",
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="Override the HuggingFace model id (e.g. Qwen/Qwen2.5-1.5B-Instruct).",
)
@click.option(
    "--lang",
    default=None,
    help="gTTS language code (default: from .env or 'en').",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream the model's output chunk-by-chunk.",
)
@click.option(
    "--show-config",
    is_flag=True,
    help="Print the resolved configuration and exit.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable debug logging.",
)
def main(
    topic: str,
    output_path: Path | None,
    play: bool,
    save: bool,
    model_id: str | None,
    lang: str | None,
    stream: bool,
    show_config: bool,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    if lang:
        settings = settings.__class__(  # type: ignore[misc]
            **{**as_dict(settings), "tts_lang": lang, "tts_slow": settings.tts_slow},
            model_id=model_id or settings.model_id,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            hf_token=settings.hf_token,
            output_dir=settings.output_dir,
        )
    elif model_id:
        settings = settings.__class__(  # type: ignore[misc]
            model_id=model_id,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            hf_token=settings.hf_token,
            prompt_word_min=settings.prompt_word_min,
            prompt_word_max=settings.prompt_word_max,
            tts_lang=settings.tts_lang,
            tts_slow=settings.tts_slow,
            output_dir=settings.output_dir,
        )

    if show_config:
        click.echo(json.dumps(as_dict(settings), indent=2))
        return

    if not topic.strip():
        raise click.UsageError("TOPIC must not be empty.")

    if output_path is None:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.output_dir / f"{slugify_topic(topic)}.mp3"

    click.echo(f"[storyteller] Loading model: {settings.model_id}", err=True)
    generator = StoryGenerator.from_defaults(settings=settings)

    click.echo(f"[storyteller] Generating story for topic: {topic!r}", err=True)
    if stream:
        story = ""
        for chunk in generator.stream(topic):
            click.echo(chunk, nl=False)
            story += chunk
        click.echo("", err=True)
    else:
        story = generator.generate(topic)
        click.echo(story)

    click.echo(
        f"[storyteller] Synthesizing with gTTS (lang={settings.tts_lang}, slow={settings.tts_slow})",
        err=True,
    )
    tts = TTS(lang=settings.tts_lang, slow=settings.tts_slow)
    audio = tts.synthesize_bytes(story)

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio)
        click.echo(f"[storyteller] Saved MP3 to {output_path}", err=True)

    if play:
        click.echo("[storyteller] Playing audio… (use --no-play to skip)", err=True)
        play_bytes(audio)

    if not save and not play:
        # The user asked for nothing to happen with the audio — be polite and
        # at least say how big the buffer is.
        click.echo(
            f"[storyteller] Audio synthesized (size={len(audio)} bytes); "
            "discarded per --no-save --no-play.",
            err=True,
        )
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()