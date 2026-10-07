"""Story generation via a HuggingFace causal language model.

This is the Python-port equivalent of the notebook's `generate_story` function.
It defaults to ``Qwen/Qwen2.5-3B-Instruct`` (the same model the notebook uses)
but any instruction-tuned causal LM that supports the ``text-generation``
``pipeline`` task will work.

Why a local model?  The original lab calls Mistral Medium through IBM
Watsonx.ai; replicating that requires IBM credentials and network access.  A
local HuggingFace model keeps the project runnable on a single laptop with no
external service in the loop, while preserving the exact prompt template and
the "story -> speech -> playback" pipeline.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .config import Settings, get_settings

# Heavy ML imports are deferred so that unit tests that don't touch the LLM
# (and tools like ``--help``) don't pay the import cost.


@dataclass(frozen=True)
class GenerationParams:
    """Mirrors the ``params`` dict used in the notebook pipeline call."""

    max_new_tokens: int = 500
    do_sample: bool = False
    max_length: int | None = None  # kept for backwards compatibility

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        return kwargs


class StoryGenerator:
    """Thin wrapper around a HuggingFace text-generation pipeline.

    Use :meth:`from_defaults` for the usual case (load the configured model
    once and reuse it for many ``generate`` calls) or construct directly when
    you already have a pipeline object you want to wrap.
    """

    def __init__(
        self,
        model_or_pipeline: Any,
        *,
        params: GenerationParams | None = None,
        settings: Settings | None = None,
    ) -> None:
        # ``model_or_pipeline`` is either a HuggingFace pipeline or a tokenizer
        # + model pair; we detect by duck-typing on ``__call__``.
        if not callable(model_or_pipeline):
            raise TypeError(
                "StoryGenerator expects a callable HuggingFace pipeline (or a "
                "compatible model object)."
            )
        self._model = model_or_pipeline
        self._params = params or GenerationParams()
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_defaults(
        cls,
        *,
        model_id: str | None = None,
        settings: Settings | None = None,
        dtype: Any | None = None,
        device_map: Any | None = None,
    ) -> "StoryGenerator":
        """Build a generator using the configured HuggingFace model.

        Imports ``torch`` and ``transformers`` lazily.  ``dtype`` defaults to
        ``torch.float16`` so the model fits on a single GPU; pass
        ``torch.float32`` for CPU-only machines.
        """
        import torch
        from transformers import pipeline

        cfg = settings or get_settings()
        chosen_model = model_id or cfg.model_id
        chosen_dtype = dtype if dtype is not None else torch.float16
        chosen_device = device_map if device_map is not None else "auto"

        kwargs: dict[str, Any] = {
            "task": "text-generation",
            "model": chosen_model,
            "dtype": chosen_dtype,
            "device_map": chosen_device,
        }
        if cfg.hf_token:
            kwargs["token"] = cfg.hf_token

        pipe = pipeline(**kwargs)
        return cls(
            pipe,
            params=GenerationParams(
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
            ),
            settings=cfg,
        )

    # ------------------------------------------------------------------ properties
    @property
    def pipeline(self) -> Any:
        """The underlying HuggingFace pipeline (escape hatch)."""
        return self._model

    @property
    def params(self) -> GenerationParams:
        return self._params

    @property
    def settings(self) -> Settings:
        return self._settings

    # ------------------------------------------------------------------ core API
    def build_prompt(self, topic: str) -> str:
        """Return the chat-formatted prompt that will be sent to the model.

        For instruction-tuned models with a chat template (Qwen, Mistral,
        Llama-3-chat, etc.) we wrap the system/user content in a one-message
        list — this is what ``pipeline(messages=...)`` consumes.  Models
        without a chat template fall back to a plain ``[INST] ... [/INST]``
        prompt.
        """
        prompt_text = self._settings.prompt(topic)
        return prompt_text

    def generate(self, topic: str, **overrides: Any) -> str:
        """Generate a story for ``topic`` and return the assistant message.

        Extra keyword arguments override :class:`GenerationParams` for this
        single call (handy for tweaking ``temperature`` etc.).
        """
        prompt_text = self.build_prompt(topic)
        messages = [{"role": "user", "content": prompt_text}]

        gen_kwargs = dict(self._params.as_kwargs())
        gen_kwargs.update(overrides)

        result = self._model(messages, **gen_kwargs)
        return _extract_assistant_message(result)

    def stream(self, topic: str, **overrides: Any) -> Iterator[str]:
        """Yield generated text chunk-by-chunk for the given ``topic``.

        Falls back to a non-streaming call if ``transformers``'s
        ``TextIteratorStreamer`` is unavailable (e.g. on an unusual model
        configuration).  Each yielded string is a non-empty chunk produced by
        the streamer.
        """
        try:
            from transformers import TextIteratorStreamer
        except ImportError:  # pragma: no cover - transformers is a hard dep
            yield self.generate(topic, **overrides)
            return

        prompt_text = self.build_prompt(topic)
        messages = [{"role": "user", "content": prompt_text}]

        # The streamer needs the tokenizer; pipelines expose it on `.tokenizer`.
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            yield self.generate(topic, **overrides)
            return

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(self._params.as_kwargs())
        gen_kwargs["streamer"] = streamer
        gen_kwargs.update(overrides)

        thread = threading.Thread(
            target=self._model, args=(messages,), kwargs=gen_kwargs
        )
        thread.start()

        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            thread.join()

    def generate_streamed(self, topic: str) -> str:
        """Convenience: run :meth:`stream` and return the joined string."""
        return "".join(self.stream(topic))


# ---------------------------------------------------------------------- helpers
def _extract_assistant_message(result: Any) -> str:
    """Pull the assistant turn out of a pipeline ``text-generation`` result.

    ``transformers.pipeline("text-generation")`` returns one of:

    * ``[{"generated_text": [{"role": ..., "content": ...}, ...]}]``
      when the input was a list of chat messages (newer transformers)
    * ``[{"generated_text": "<prompt>\\n<continuation>"}]``
      when the input was a raw string
    """
    if not result:
        return ""
    first = result[0]
    text = first.get("generated_text", first)

    # Chat-message list form.
    if isinstance(text, list) and text and isinstance(text[-1], dict):
        last = text[-1]
        return str(last.get("content", "")).strip()

    # Raw-string form: strip the prompt prefix if the pipeline echoed it.
    if isinstance(text, str):
        return text.strip()
    return str(text)


# ``generate_story`` is the name from the notebook; keep it as an alias so the
# notebook->script port reads naturally.
def generate_story(generator: StoryGenerator, topic: str) -> str:
    """Notebook-style convenience wrapper around :meth:`StoryGenerator.generate`."""
    return generator.generate(topic)


__all__ = ["GenerationParams", "StoryGenerator", "generate_story"]


# Silence "imported but unused" for typing-only names on older Pythons.
_ = Callable