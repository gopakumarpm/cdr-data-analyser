# Anime Line Art Studio

A Streamlit app that turns text prompts into downloadable anime-style black-and-white line art.

## Features
- Prompt-based line art generation
- Manga-inspired composition (character + scene cues)
- Deterministic outputs for the same prompt
- Adjustable resolution, line thickness, and detail level
- One-click PNG download

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Usage
1. Enter a scene description (for example: `An anime hero in a moonlit city street with speed lines`).
2. Adjust canvas size and style controls.
3. Click **Generate Anime Line Art**.
4. Click **Download PNG** to save the output.

## Notes
- Scene keywords like `city`, `night`, `moon`, `action`, and `fight` influence the generated composition.
- Output is generated locally; no external model/API is required.
