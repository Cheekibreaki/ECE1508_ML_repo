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
N_IMAGES: int = 14000
START_ROW: int = 14_000
TRAIN_CSV: str = "fashion-mnist_train.csv"
OUT_DIR_TMPL: str = "data_generation_{start}-{end}_{timestamp}"
MAX_TOKENS: int = 30

# Provide your API key via env‑var or replace placeholder
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")



MAX_DESC_WORDS: int = 20

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

def load_data(csv_path: str, n_images: int, start_row: int = 0):
    """Load *n_images* rows beginning at *start_row* from the CSV."""
    df = pd.read_csv(csv_path)
    df = df.iloc[start_row : start_row + n_images]   # ← skip the first rows
    labels = df["label"].tolist()
    pixels_df = df.drop(columns=["label"])
    return pixels_df, labels

def _is_valid_desc(desc: str, max_words: int = MAX_DESC_WORDS) -> bool:
    """True if `desc` has ≤ `max_words` words."""
    if len(desc.strip().split()) <= max_words:
        return True
    else:
        print(desc)
        return False

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
    pixels_df, labels = load_data(TRAIN_CSV, N_IMAGES, start_row=START_ROW)

    # 2. Prepare output directory
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        OUT_DIR_TMPL.format(
            start=START_ROW,
            end=START_ROW + N_IMAGES - 1,
            timestamp=timestamp,
        )
    )
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


        # prompt = (  f"""
        # You are a product copywriter creating concise catalog-friendly descriptors for Fashion-MNIST items.
        # Image label: '{LABEL_MAP[label_id]}'.
        # Generate a refined 4–5 WORD English descriptor using elevated, tasteful adjectives and style/fabric nouns (no verbs).
        # Do NOT include any explicit category terms (T-shirt, Trouser, etc.) in the descriptor body.
        # Append a single space and then the category label exactly as given: {LABEL_MAP[label_id]}.
        # Return ONE LINE ONLY, suitable for a filename (letters, numbers, spaces only; no punctuation).
        # Example format: "textured minimalist knit panel {LABEL_MAP[label_id]}".
        # Image follows.
        # """
        #             )
        # prompt = (f"""
        #     You are describing a Fashion-MNIST image for a dataset.
        #     Label: '{LABEL_MAP[label_id]}'.
        #     Write ONE short grammatical English sentence (up to ~12 words) that naturally describes the item. Everyday tone is fine.
        #     Avoid saying the category name inside the sentence (no T-shirt, Trouser, etc.).
        #     After the sentence, append a single space and then the category label exactly as given: {LABEL_MAP[label_id]}.
        #     Return ONE LINE ONLY. Minimal punctuation: end the sentence with a period before the space+category is OK, or omit punctuation entirely; just be consistent.
        #     Example format: "A soft knit top with relaxed shoulders. {LABEL_MAP[label_id]}".
        #     Image follows.
        #     """)

        prompt = (
            f"The following greyscale image is a Fashion‑MNIST item labeled ‘{LABEL_MAP[label_id]}’.\n"
            "Return a 4–5 word English description in simple, everyday language.\n"
            "Use only adjectives and nouns (e.g., 'soft cotton casual wear').\n"
            "Do not use category words like T‑shirt or Trouser.\n"
            "Add the original category word (‘{LABEL_MAP[label_id]}’) at the end."
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
            print(f"[WARN] OpenAI failed for idx={idx}: {exc}. Skipping this image.")
            continue  # ⬅ Skip this image completely

        if not description or not _is_valid_desc(description):
            print(f"[WARN] Invalid description for idx={idx}: '{description}'. Skipping.")
            continue  # ⬅ Skip this image completely

        filename = _sanitize_filename(description) + ".png"
        img.save(out_dir / filename)

    print(f"Saved {N_IMAGES} images to → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
