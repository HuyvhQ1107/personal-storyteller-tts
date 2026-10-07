# Use Mistral and gTTS to Create Your Personal Storyteller

A small Python project that turns any topic into a short, engaging, spoken-word story.

It is the Python-application version of the Jupyter notebook
[`Use_Mistral_and_gTTS_to_Create_Your_Personal_Storyteller.ipynb`](./Use_Mistral_and_gTTS_to_Create_Your_Personal_Storyteller.ipynb).
The notebook mixes Watsonx AI's Mistral with `gTTS`; here we swap in a local
`HuggingFace` causal LM that you can run on your own hardware — the default is
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (the same
model the notebook uses in practice) and the rest of the pipeline is unchanged.

## How it works

```
topic ─▶ StoryGenerator (local LLM) ─▶ story ─▶ gTTS ─▶ MP3 bytes ─▶ player
```

1. **Story generation** – a small instruction-tuned LLM writes a 200–500 word
   educational story for beginners, with a friendly tone and a short summary.
2. **Speech synthesis** – `gTTS` (Google Text-to-Speech) turns the text into an
   MP3, optionally written to disk or played through the speakers.
3. **Playback** – the in-memory MP3 is streamed back through `pygame` so you
   hear the story without ever leaving the terminal.

## Project layout

```
d:\TTS\
├── personal_storyteller\      # The Python package
│   ├── __init__.py
│   ├── config.py              # Env-driven configuration
│   ├── storyteller.py         # LLM-backed StoryGenerator
│   ├── tts.py                 # gTTS wrapper (file or bytes)
│   ├── player.py              # pygame-based playback
│   └── cli.py                 # `storyteller` console script
├── examples\
│   └── generate_demo.py       # Quick programmatic demo
├── output\                    # Generated MP3 files land here
├── tests\
│   └── test_smoke.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── Use_Mistral_and_gTTS_to_Create_Your_Personal_Storyteller.ipynb   # original
```

## Installation

1. Create a virtual environment (optional but recommended) and activate it:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies. `torch` is large; install it with the correct CUDA /
   CPU index URL for your hardware if needed.

   ```powershell
   pip install -r requirements.txt
   ```

3. (Optional) Copy the configuration template:

   ```powershell
   copy .env.example .env
   ```

   Then edit `.env` to point at a different model, change the TTS language, etc.

## Quick start

### Command-line

```powershell
# Generate a story about butterflies, save to output\butterflies.mp3, play it.
python -m personal_storyteller.cli "the life cycle of butterflies" -o output\butterflies.mp3 --play

# Don't play, just save.
python -m personal_storyteller.cli "romantic relationships" -o output\love.mp3

# Print the story only.
python -m personal_storyteller.cli "the discovery of penicillin" --no-save --no-play
```

After `pip install -e .`, the `storyteller` console script is also available:

```powershell
storyteller "the discovery of penicillin" -o output\penicillin.mp3 --play
```

### Programmatic

```python
from personal_storyteller import StoryGenerator, TTS, play_bytes, save_bytes

generator = StoryGenerator.from_defaults()     # loads Qwen2.5-3B-Instruct
story = generator.generate("the life cycle of butterflies")
print(story)

audio = TTS(lang="en").synthesize_bytes(story)
save_bytes(audio, "output/butterflies.mp3")
play_bytes(audio)
```

### Demo script

```powershell
python examples\generate_demo.py
```

## Configuration

All settings come from environment variables (or a local `.env` file). The full
list lives in [`personal_storyteller/config.py`](./personal_storyteller/config.py).

| Variable                       | Meaning                                       |
|-------------------------------|----------------------------------------------|
| `STORYTELLER_MODEL_ID`        | HuggingFace model id (default `Qwen/Qwen2.5-3B-Instruct`) |
| `STORYTELLER_MAX_NEW_TOKENS`  | Max tokens to generate (default `500`)      |
| `STORYTELLER_DO_SAMPLE`       | `true` / `false` (default `false`)           |
| `STORYTELLER_TTS_LANG`        | gTTS language code (default `en`)            |
| `STORYTELLER_TTS_SLOW`        | `true` / `false` (default `false`)           |
| `STORYTELLER_OUTPUT_DIR`      | Where MP3 files are written (default `output`) |
| `HF_TOKEN`                    | HuggingFace token for gated models (optional) |

## Hardware

- The default 3B model fits comfortably on a single 8 GB GPU (and even on
  modern CPUs if you're patient). Switch to a smaller model — for example
  `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-0.5B-Instruct` — for low-RAM
  machines.
- Use a CUDA-enabled `torch` install for the fastest experience. CPU-only is
  supported but slow.

## Tests

A tiny smoke test guards the prompt assembly and the gTTS synthesizer (it
does not load the LLM, so it stays fast and CI-friendly):

```powershell
pytest -q
```

## Acknowledgements

Adapted from the IBM Skills Network guided project
**"Use Mistral and gTTS to Create Your Personal Storyteller"**, originally
delivered on the Watsonx.ai platform. The notebook is kept in this repository
for reference.