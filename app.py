"""Anime-style line art generator from text prompts."""

from __future__ import annotations

import hashlib
import io
import random

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

st.set_page_config(page_title="Anime Line Art Studio", page_icon="🖌️", layout="wide")


st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #1f1147 0%, #5f2eea 55%, #8f77ff 100%);
        border-radius: 18px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; }
    .hero p { margin-bottom: 0; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Anime Line Art Studio</h1>
      <p>Describe your scene, generate clean manga-style line art, and download instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _draw_character(draw: ImageDraw.ImageDraw, w: int, h: int, rng: random.Random, lw: int) -> None:
    cx, cy = w * 0.5, h * 0.55
    head_r = min(w, h) * rng.uniform(0.12, 0.18)
    draw.ellipse((cx - head_r, cy - head_r, cx + head_r, cy + head_r), outline=0, width=lw)

    # Eyes
    eye_y = cy - head_r * 0.1
    eye_dx = head_r * 0.4
    eye_w = head_r * 0.32
    eye_h = head_r * 0.18
    for side in (-1, 1):
        ex = cx + side * eye_dx
        draw.arc((ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h), 200, 340, fill=0, width=lw)
        draw.ellipse((ex - eye_w * 0.3, eye_y - eye_h * 0.1, ex + eye_w * 0.3, eye_y + eye_h * 0.2), outline=0, width=max(1, lw - 1))

    # Hair spikes
    for i in range(16):
        angle = np.deg2rad(180 + i * 11 + rng.uniform(-5, 5))
        sx = cx + np.cos(angle) * head_r * 0.7
        sy = cy + np.sin(angle) * head_r * 0.7
        ex = cx + np.cos(angle) * head_r * rng.uniform(1.2, 1.8)
        ey = cy + np.sin(angle) * head_r * rng.uniform(1.1, 1.7)
        draw.line((sx, sy, ex, ey), fill=0, width=max(1, lw - 1))

    # Body
    torso_top = cy + head_r * 0.9
    draw.line((cx, torso_top, cx, torso_top + head_r * 1.7), fill=0, width=lw)
    draw.line((cx, torso_top + head_r * 0.4, cx - head_r, torso_top + head_r), fill=0, width=lw)
    draw.line((cx, torso_top + head_r * 0.4, cx + head_r, torso_top + head_r), fill=0, width=lw)


def _draw_scene(draw: ImageDraw.ImageDraw, prompt: str, w: int, h: int, rng: random.Random, lw: int) -> None:
    lower = prompt.lower()
    if any(word in lower for word in ("city", "tokyo", "street", "building")):
        x = 0
        while x < w:
            bw = rng.randint(max(24, w // 18), max(40, w // 10))
            bh = rng.randint(h // 5, h // 2)
            draw.rectangle((x, h - bh, min(w, x + bw), h), outline=0, width=lw)
            for wx in range(x + 6, min(w, x + bw - 4), 10):
                for wy in range(h - bh + 8, h - 6, 12):
                    draw.rectangle((wx, wy, wx + 4, wy + 5), outline=0, width=1)
            x += bw + rng.randint(6, 14)

    if any(word in lower for word in ("moon", "night", "sky", "stars")):
        r = min(w, h) * 0.08
        mx, my = w * 0.82, h * 0.18
        draw.ellipse((mx - r, my - r, mx + r, my + r), outline=0, width=lw)
        draw.arc((mx - r * 0.65, my - r * 0.95, mx + r * 0.95, my + r * 0.95), 70, 290, fill=255, width=lw + 2)
        for _ in range(40):
            sx, sy = rng.randint(0, w - 1), rng.randint(0, int(h * 0.55))
            draw.point((sx, sy), fill=0)

    if any(word in lower for word in ("action", "fight", "energy", "speed")):
        for _ in range(120):
            x0 = rng.randint(0, w)
            y0 = rng.randint(0, h)
            length = rng.randint(w // 15, w // 5)
            draw.line((x0, y0, x0 + length, y0 - length // 5), fill=0, width=max(1, lw - 1))


def generate_line_art(prompt: str, width: int, height: int, line_thickness: int, detail_level: int) -> Image.Image:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = random.Random(seed)

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # Manga panel frame
    margin = max(10, width // 45)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=0, width=line_thickness)

    # Primary character + scene
    _draw_character(draw, width, height, rng, line_thickness)
    _draw_scene(draw, prompt, width, height, rng, line_thickness)

    # Add prompt-based abstract contours to make each result unique
    for i, ch in enumerate(prompt[: detail_level * 22]):
        val = (ord(ch) + i * 7) % 255
        x = (i * 19 + val * 3) % width
        y = (i * 13 + val * 5) % height
        radius = 8 + (val % (10 + detail_level * 2))
        draw.arc((x - radius, y - radius, x + radius, y + radius), 10, 250, fill=0, width=max(1, line_thickness - 1))

    # Edge-extract + threshold for clean line-art finish
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.8))
    edges = ImageEnhance.Contrast(edges).enhance(2.4)

    arr = np.array(edges)
    binary = np.where(arr > 42, 0, 255).astype(np.uint8)
    output = Image.fromarray(binary, mode="L")
    return output


left, right = st.columns([1, 1])

with left:
    prompt = st.text_area(
        "Describe your anime scene",
        value="A lone anime swordsman in a neon Tokyo street at night with dramatic speed lines",
        height=150,
        help="Include words like city, moon, action, fight, night, sky to influence the drawing.",
    )

    col1, col2 = st.columns(2)
    with col1:
        width = st.slider("Width", 512, 1536, 1024, step=128)
        line_thickness = st.slider("Line thickness", 1, 6, 3)
    with col2:
        height = st.slider("Height", 512, 1536, 1024, step=128)
        detail_level = st.slider("Detail level", 1, 10, 6)

    generate = st.button("Generate Anime Line Art", use_container_width=True)

with right:
    st.subheader("Output Preview")
    if "output_image" not in st.session_state:
        st.session_state.output_image = None

    if generate and prompt.strip():
        st.session_state.output_image = generate_line_art(
            prompt=prompt.strip(),
            width=width,
            height=height,
            line_thickness=line_thickness,
            detail_level=detail_level,
        )

    if st.session_state.output_image is not None:
        st.image(st.session_state.output_image, width="stretch", caption="Generated manga-style line art")

        buffer = io.BytesIO()
        st.session_state.output_image.save(buffer, format="PNG")
        st.download_button(
            label="Download PNG",
            data=buffer.getvalue(),
            file_name="anime_line_art.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.info("Generate an image to preview and download it.")
