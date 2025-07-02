# fashionmnist_description_generator.py
"""Generate 4‑5‑word descriptions for the first 100 Fashion‑MNIST training images,
then save each image with the new description as its filename in a dated folder.

This version **loads the data via a `load_data` helper** that mirrors the user‑supplied
CSV‑based approach instead of KaggleHub. Place the original Fashion‑MNIST CSV files
here:
    data_sampling/fashion-mnist_train.csv

Requirements
------------
    pip install pillow pandas openai tqdm

Run
---
    OPENAI_API_KEY=... python fashionmnist_description_generator.py
"""
from __future__ import annotations

import base64
import datetime as _dt
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from openai import OpenAI, OpenAIError

# ─────────────────────────── Configuration ──────────────────────────
N_IMAGES: int = 100
TRAIN_CSV: str = "fashion-mnist_train.csv"
OUT_DIR_TMPL: str = "data_generation_{start}-{end}_{date}"
MAX_TOKENS: int = 20

# Provide your API key via env‑var or replace placeholder
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

LABEL_MAP = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# ────────────────────────────── Helpers ─────────────────────────────

def load_data(csv_path: str, n_images: int) -> Tuple[pd.DataFrame, List[int]]:
    """Load *n_images* rows from the local Fashion‑MNIST CSV.

    Mimics the structure of the user‑provided `load_data` function but focuses on
    the **training set only** because we just need sample images. Returns a data
    frame (`pixels_df`) and the corresponding *label* list.
    """
    df = pd.read_csv(csv_path)
    df = df.head(n_images)
    labels = df["label"].tolist()
    pixels_df = df.drop(columns=["label"])
    return pixels_df, labels


def _sanitize_filename(text: str) -> str:
    """Convert arbitrary text to a safe, lowercase filename."""
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "image"


def _image_url_obj(img: Image.Image, detail: str = "auto") -> Dict[str, Any]:
    """Return the dict expected by OpenAI vision: {"url": ..., "detail": ...}."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def main() -> None:
    # 1. Load the first *N_IMAGES* images & labels via user‑style loader
    pixels_df, labels = load_data(TRAIN_CSV, N_IMAGES)

    # 2. Prepare output directory
    today = _dt.date.today().isoformat()
    out_dir = Path(OUT_DIR_TMPL.format(start=0, end=N_IMAGES - 1, date=today))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Initialise OpenAI client
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY environment variable or hard‑code it in the script.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # 4. Generate captions & save images one by one
    for idx, (label_id, pixel_row) in tqdm(
        enumerate(zip(labels, pixels_df.values)), total=N_IMAGES, desc="Processing"
    ):
        # Convert flat pixel array to 28×28 grayscale PIL image
        img = Image.fromarray(pixel_row.reshape(28, 28).astype("uint8"), mode="L")

        prompt = (
            f"The following greyscale image is a Fashion‑MNIST item labeled ‘{LABEL_MAP[label_id]}’.\n"
            "Return a catchy 4–5‑word English description suitable as a filename.\n"
            "Avoid category words like T‑shirt, Trouser, etc.; use only descriptive adjectives/nouns."
        )

        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",  "text": prompt},
                            {"type": "input_image", "image_url": _image_url_obj(img)},
                        ],
                    }
                ],
            )
            description: str = response.output_text
        except (OpenAIError, KeyError, IndexError) as exc:
            print(f"[WARN] OpenAI failed for idx={idx}: {exc}. Using fallback name.")
            description = f"item_{idx}"

        filename = _sanitize_filename(description) + ".png"
        img.save(out_dir / filename)

    print(f"Saved {N_IMAGES} images to → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
